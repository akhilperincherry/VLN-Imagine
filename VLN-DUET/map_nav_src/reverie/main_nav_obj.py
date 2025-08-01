import os
import json
import time
import numpy as np
from collections import defaultdict

import torch
from tensorboardX import SummaryWriter

from utils.misc import set_random_seed
from utils.logger import write_to_record_file, print_progress, timeSince
from utils.distributed import init_distributed, is_default_gpu
from utils.distributed import all_gather, merge_dist_results

from models.vlnbert_init import get_tokenizer

from utils.data import ImageFeaturesDB

from reverie.agent_obj import GMapObjectNavAgent
from reverie.data_utils import ObjectFeatureDB, construct_instrs, load_obj2vps, ImaginationImageFeaturesDB
from reverie.env import ReverieObjectNavBatch
from reverie.parser import parse_args


def build_dataset(args, rank=0, is_test=False):
    tok = get_tokenizer(args)

    feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)
    obj_db = ObjectFeatureDB(args.obj_ft_file, args.obj_feat_size)
    obj2vps = load_obj2vps(os.path.join(args.anno_dir, 'BBoxes.json'))

    if args.imagine_enc_pano:
        feat_imag_train_db = ImaginationImageFeaturesDB(args.imagination_train_ft_file, args.image_feat_size)
        feat_imag_val_seen_db = ImaginationImageFeaturesDB(args.imagination_val_seen_ft_file, args.image_feat_size)
        feat_imag_val_unseen_db = ImaginationImageFeaturesDB(args.imagination_val_unseen_ft_file, args.image_feat_size)

        if is_test:
            feat_imag_test_db = ImaginationImageFeaturesDB(args.imagination_test_ft_file, args.image_feat_size)
            feat_imagine_dict = {'train': feat_imag_train_db, 'val_seen': feat_imag_val_seen_db, 'val_unseen': feat_imag_val_unseen_db, 'test': feat_imag_test_db}
        else:
            feat_imagine_dict = {'train': feat_imag_train_db, 'val_seen': feat_imag_val_seen_db, 'val_unseen': feat_imag_val_unseen_db}

    else:
        feat_imagine_dict = None

    dataset_class = ReverieObjectNavBatch

    # because we don't use distributed sampler here
    # in order to make different processes deal with different training examples
    # we need to shuffle the data with different seed in each processes
    if args.aug is not None:
        aug_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [args.aug], 
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len
        )
        aug_env = dataset_class(
            feat_db, obj_db, aug_instr_data, args.connectivity_dir, obj2vps, 
            batch_size=args.batch_size, max_objects=args.max_objects,
            angle_feat_size=args.angle_feat_size, 
            seed=args.seed+rank, sel_data_idxs=None, name='aug', 
            multi_endpoints=args.multi_endpoints, multi_startpoints=args.multi_startpoints,
        )
    else:
        aug_env = None

    if args.aug_only:
        train_env, aug_env = aug_env, None
        args.aug = None
    else:
        train_instr_data = construct_instrs(
            args.anno_dir, args.dataset, ['train'], 
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len
        )
        train_env = dataset_class(
            feat_db, obj_db, train_instr_data, args.connectivity_dir, obj2vps,
            batch_size=args.batch_size, max_objects=args.max_objects,
            angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
            sel_data_idxs=None, name='train', 
            multi_endpoints=args.multi_endpoints, multi_startpoints=args.multi_startpoints, imagine_feat_dict = feat_imagine_dict
        )
        if args.imagine_enc_pano == False:
            assert train_env.imagine_feat_dict == None

    val_env_names = ['val_train_seen', 'val_seen', 'val_unseen']

    if args.submit:
        val_env_names.append('test')
        
    val_envs = {}
    for split in val_env_names:
        val_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [split], 
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len
        )
        val_env = dataset_class(
            feat_db, obj_db, val_instr_data, args.connectivity_dir, obj2vps, batch_size=args.batch_size, 
            angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
            sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name=split,
            max_objects=None, multi_endpoints=False, multi_startpoints=False, imagine_feat_dict=feat_imagine_dict
        )   # evaluation using all objects
        val_envs[split] = val_env

    return train_env, val_envs, aug_env


def train(args, train_env, val_envs, aug_env=None, rank=-1):
    default_gpu = is_default_gpu(args)

    if default_gpu:
        with open(os.path.join(args.log_dir, 'training_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        writer = SummaryWriter(log_dir=args.log_dir)
        record_file = os.path.join(args.log_dir, 'train.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    agent_class = GMapObjectNavAgent
    listner = agent_class(args, train_env, rank=rank)

    # resume file
    start_iter = 0
    if args.resume_file is not None:
        start_iter = listner.load(os.path.join(args.resume_file))
        if default_gpu:
            write_to_record_file(
                "\nLOAD the model from {}, iteration ".format(args.resume_file, start_iter),
                record_file
            )
        start_iter = 0 # To avoid plots being offset by start_iter when resuming from a model, which is not important in our case.
       
    # first evaluation
    if args.eval_first:
        loss_str = "validation before training"
        for env_name, env in val_envs.items():
            print(env_name)
            listner.env = env
            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            preds = listner.get_results()
            # gather distributed results
            preds = merge_dist_results(all_gather(preds))
            if default_gpu:
                score_summary, _ = env.eval_metrics(preds)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
        if default_gpu:
            write_to_record_file(loss_str, record_file)

    start = time.time()
    if default_gpu:
        write_to_record_file(
            '\nListener training starts, start iteration: %s' % str(start_iter), record_file
        )

    best_val = {'val_unseen': {"spl": 0., "sr": 0., "state":""}}

    for idx in range(start_iter, start_iter+args.iters, args.log_every):
        listner.logs = defaultdict(list)
        interval = min(args.log_every, args.iters-idx)
        iter = idx + interval

        if args.imagine_enc_pano:

            #variant 4
            if args.experimental_warmup and args.experimental_warmup_type == 'variant4':

                contrastive_param_lrs = {'stage1': args.lr * 10, 'stage2': args.lr * 5, 'stage3': args.lr * 0.1}
                bert_model_param_lrs = {'stage1': None, 'stage2': args.lr * 0.1, 'stage3': args.lr * 0.1}
                contrastive_param_trainable = {'stage1': True, 'stage2': True, 'stage3': True}
                bert_param_trainable = {'stage1': False, 'stage2': True, 'stage3': True}

                assert len(listner.vln_bert_optimizer.param_groups) == 3, 'Expected 3 optimizer groups for contastive params, imagination parms, and rest of vln_bert params.'

                if idx < ((0.25 * args.iters) + start_iter):
                    warm_up_stage = 'stage1'
                    # Freeze all parameters except for newly introduced.
                    for param in listner.vln_bert.parameters():
                        param.requires_grad = bert_param_trainable[warm_up_stage]
                    for param in listner.vln_bert.vln_bert.contrastive_alignment_model.parameters():
                        param.requires_grad = contrastive_param_trainable[warm_up_stage]
                    for param in listner.vln_bert.vln_bert.imagine_embeddings.parameters():
                        param.requires_grad = contrastive_param_trainable[warm_up_stage]
                    listner.vln_bert_optimizer.param_groups[0]['lr'] = contrastive_param_lrs[warm_up_stage]  # contrastive_alignment_model
                    listner.vln_bert_optimizer.param_groups[1]['lr'] = contrastive_param_lrs[warm_up_stage]  # imagine_embeddings
                
                # Mid-phase: set moderate learning rates for both specific and main parameters
                elif idx >= ((0.25 * args.iters) + start_iter) and idx < ((0.5 * args.iters) + start_iter):
                    warm_up_stage = 'stage2'
                    for param in listner.vln_bert.parameters():
                        param.requires_grad = True
                    listner.vln_bert_optimizer.param_groups[2]['lr'] = bert_model_param_lrs[warm_up_stage]  # main_vln_bert_params
                    listner.vln_bert_optimizer.param_groups[0]['lr'] = contrastive_param_lrs[warm_up_stage]  # contrastive_alignment_model
                    listner.vln_bert_optimizer.param_groups[1]['lr'] = contrastive_param_lrs[warm_up_stage]  # imagine_embeddings
                                    
                # Final phase: set all learning rates back to base rate
                else:                    
                    warm_up_stage = 'stage3'
                    assert bert_model_param_lrs[warm_up_stage] == contrastive_param_lrs[warm_up_stage]
                    for param_group in listner.vln_bert_optimizer.param_groups:
                        param_group['lr'] = bert_model_param_lrs[warm_up_stage]

        # Train for log_every interval
        if aug_env is None:
            listner.env = train_env
            listner.train(interval, feedback=args.feedback)  # Train interval iters
        else:
            jdx_length = len(range(interval // 2))
            for jdx in range(interval // 2):
                # Train with GT data
                listner.env = train_env
                listner.train(1, feedback=args.feedback)

                # Train with Augmented data
                listner.env = aug_env
                listner.train(1, feedback=args.feedback)

                if default_gpu:
                    print_progress(jdx, jdx_length, prefix='Progress:', suffix='Complete', bar_length=50)

        if default_gpu:
            # Log the training stats to tensorboard
            total = max(sum(listner.logs['total']), 1)          # RL: total valid actions for all examples in the batch
            length = max(len(listner.logs['critic_loss']), 1)   # RL: total (max length) in the batch
            critic_loss = sum(listner.logs['critic_loss']) / total
            policy_loss = sum(listner.logs['policy_loss']) / total
            OG_loss = sum(listner.logs['OG_loss']) / max(len(listner.logs['OG_loss']), 1)
            IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)
            cosine_loss = sum(listner.logs['contrastive_loss']) / max(len(listner.logs['contrastive_loss']), 1)
            entropy = sum(listner.logs['entropy']) / total
            writer.add_scalar("loss/critic", critic_loss, idx)
            writer.add_scalar("policy_entropy", entropy, idx)
            writer.add_scalar("loss/OG_loss", OG_loss, idx)
            writer.add_scalar("loss/IL_loss", IL_loss, idx)
            writer.add_scalar("loss/cosine_loss", cosine_loss, idx)
            writer.add_scalar("total_actions", total, idx)
            writer.add_scalar("max_length", length, idx)
            write_to_record_file(
                "\ntotal_actions %d, max_length %d, entropy %.4f, IL_loss %.4f, OG_loss %.4f, policy_loss %.4f, critic_loss %.4f, cosine_loss %.4f" % (
                    total, length, entropy, IL_loss, OG_loss, policy_loss, critic_loss, cosine_loss),
                record_file
            )

        # Run validation
        loss_str = "iter {}".format(iter)
        for env_name, env in val_envs.items():
            listner.env = env

            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            preds = listner.get_results()
            preds = merge_dist_results(all_gather(preds))

            if default_gpu:
                score_summary, _ = env.eval_metrics(preds)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                    writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], idx)

                # select model by spl
                if env_name in best_val:
                    if iter%2000==0:
                        os.makedirs(os.path.join(args.ckpt_dir, 'all_ckpts'), exist_ok=True)
                        listner.save(idx, os.path.join(args.ckpt_dir, 'all_ckpts', "iter_%d_SR_%f_SPL_%f_%s" % (iter, np.round(score_summary['sr'], 2), np.round(score_summary['spl'], 2), env_name)))

                    if score_summary['spl'] >= best_val[env_name]['spl']:
                        best_val[env_name]['spl'] = score_summary['spl']
                        best_val[env_name]['sr'] = score_summary['sr']
                        best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                        listner.save(idx, os.path.join(args.ckpt_dir, "best_%s" % (env_name)))
                
        
        if default_gpu:
            listner.save(idx, os.path.join(args.ckpt_dir, "latest_dict"))

            write_to_record_file(
                ('%s (%d %d%%) %s' % (timeSince(start, float(iter)/args.iters), iter, float(iter)/args.iters*100, loss_str)),
                record_file
            )
            write_to_record_file("BEST RESULT TILL NOW", record_file)
            for env_name in best_val:
                write_to_record_file(env_name + ' | ' + best_val[env_name]['state'], record_file)


def valid(args, train_env, val_envs, rank=-1):
    default_gpu = is_default_gpu(args)

    agent_class = GMapObjectNavAgent
    agent = agent_class(args, train_env, rank=rank)

    if args.resume_file is not None:
        print("Loaded the listener model at iter %d from %s" % (
            agent.load(args.resume_file), args.resume_file))

    if default_gpu:
        with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(args.log_dir, 'valid.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    for env_name, env in val_envs.items():
        prefix = 'submit' if args.detailed_output is False else 'detail'
        output_file = os.path.join(args.pred_dir, "%s_%s_%s.json" % (
            prefix, env_name, args.fusion))
        if os.path.exists(output_file):
            continue
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        start_time = time.time()
        agent.test(
            use_dropout=False, feedback='argmax', iters=iters)
        print(env_name, 'cost time: %.2fs' % (time.time() - start_time))
        preds = agent.get_results(detailed_output=args.detailed_output)
        preds = merge_dist_results(all_gather(preds))

        if default_gpu:
            if 'test' not in env_name:
                score_summary, score_individual = env.eval_metrics(preds)
                loss_str = "Env name: %s" % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                write_to_record_file(loss_str+'\n', record_file)

                json.dump(
                    score_individual,
                    open(os.path.join(args.pred_dir, "individual_metrics_%s.json" % env_name), 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )


            if args.submit:
                json.dump(
                    preds, open(output_file, 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )
                


def main():
    args = parse_args()

    if args.world_size > 1:
        rank = init_distributed(args)
        torch.cuda.set_device(args.local_rank)
    else:
        rank = 0

    set_random_seed(args.seed + rank)

    if not args.test:
        train_env, val_envs, aug_env = build_dataset(args, rank=rank)
    else:
        train_env, val_envs, aug_env = build_dataset(args, rank=rank, is_test=True)

    if not args.test:
        print('Beginning Training')
        train(args, train_env, val_envs, aug_env=aug_env, rank=rank)
    else:
        print('Beginning Testing')
        valid(args, train_env, val_envs, rank=rank)
            

if __name__ == '__main__':
    main()
