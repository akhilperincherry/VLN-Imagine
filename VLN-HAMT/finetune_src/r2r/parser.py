import argparse
import os
import torch
import json
import jsonlines


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--root_dir', type=str, default='../datasets')
    parser.add_argument('--root_dir_fgr2r', type=str, default='/nfs/hpc/sw/perincha/repos/Fine-Grained-R2R')

    parser.add_argument(
        '--dataset', type=str, default='r2r', 
        choices=['r2r', 'r4r', 'r2r_back', 'r2r_last', 'rxr']
    )
    parser.add_argument('--langs', nargs='+', default=None, choices=['en', 'hi', 'te'])
    parser.add_argument('--output_dir', type=str, default='default', help='experiment id')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tokenizer', choices=['bert', 'xlm'], default='bert')

    # distributional training (single-node, multiple-gpus)
    parser.add_argument('--world_size', type=int, default=1, help='number of gpus')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument("--node_rank", type=int, default=0, help="Id of the node")
    
    # General
    parser.add_argument('--iters', type=int, default=300000, help='training iterations')
    parser.add_argument('--log_every', type=int, default=2000)
    parser.add_argument('--eval_first', action='store_true', default=False)
        
    parser.add_argument('--ob_type', type=str, choices=['cand', 'pano'], default='pano')
    parser.add_argument('--test', action='store_true', default=False)

    # Data preparation
    parser.add_argument('--max_instr_len', type=int, default=80)
    parser.add_argument('--max_action_len', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--ignoreid', type=int, default=-100, help='ignoreid for action')
    
    # Load the model from
    parser.add_argument("--resume_file", default=None, help='path of the trained model')
    parser.add_argument("--resume_optimizer", action="store_true", default=False)

    # Augmented Paths from
    parser.add_argument("--aug", default=None)
    parser.add_argument('--bert_ckpt_file', default=None, help='init vlnbert')

    # Listener Model Config
    parser.add_argument("--ml_weight", type=float, default=0.20)
    parser.add_argument('--entropy_loss_weight', type=float, default=0.01)
    parser.add_argument("--teacher_weight", type=float, default=1.)

    parser.add_argument("--features", type=str, default='places365')
    parser.add_argument('--fix_lang_embedding', action='store_true', default=False)
    parser.add_argument('--fix_hist_embedding', action='store_true', default=False)
    parser.add_argument('--fix_obs_embedding', action='store_true', default=False)

    parser.add_argument('--num_l_layers', type=int, default=9)
    parser.add_argument('--num_h_layers', type=int, default=0)
    parser.add_argument('--num_x_layers', type=int, default=4)
    parser.add_argument('--hist_enc_pano', action='store_true', default=False)
    parser.add_argument('--hist_pano_num_layers', type=int, default=2)
    # cmt
    parser.add_argument('--no_lang_ca', action='store_true', default=False)
    parser.add_argument('--act_pred_token', default='ob_txt', choices=['ob', 'ob_txt', 'ob_hist', 'ob_txt_hist', 'ob_imagine_text'])

    # Dropout Param
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--feat_dropout', type=float, default=0.3)

    # Submision configuration
    parser.add_argument("--submit", action='store_true', default=False)
    parser.add_argument('--no_cand_backtrack', action='store_true', default=False)

    # Training Configurations
    parser.add_argument(
        '--optim', type=str, default='rms',
        choices=['rms', 'adam', 'adamW', 'sgd']
    )    # rms, adam
    parser.add_argument('--lr', type=float, default=0.00001, help="the learning rate")
    parser.add_argument('--decay', dest='weight_decay', type=float, default=0.)
    parser.add_argument(
        '--feedback', type=str, default='sample',
        help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``'
    )
    parser.add_argument(
        '--teacher', type=str, default='final',
        help="How to get supervision. one of ``next`` and ``final`` "
    )
    parser.add_argument('--epsilon', type=float, default=0.1, help='')

    # Model hyper params:
    parser.add_argument("--angle_feat_size", type=int, default=4)
    parser.add_argument('--image_feat_size', type=int, default=2048)
    parser.add_argument('--views', type=int, default=36)

    # A2C
    parser.add_argument("--gamma", default=0.9, type=float, help='reward discount factor')
    parser.add_argument(
        "--normalize", dest="normalize_loss", default="total", 
        type=str, help='batch or total'
    )

    # Imagine trajs. 
    parser.add_argument('--imagine_enc_pano', action='store_true', default=False)
    parser.add_argument('--imagination_data_v2', action='store_true', default=False)
    parser.add_argument('--bypass_imag_encoder', action='store_true', default=False) #Pass imag features directly to CMT.
    parser.add_argument('--fix_imagine_embeds', action='store_true', default=False)
    parser.add_argument('--max_imagination_len', type=int, default=25)
    parser.add_argument('--experimental_warmup', action='store_true', default=False)
    parser.add_argument('--experimental_warmup_type', default='variant1', choices=['variant4'])
    parser.add_argument('--no_loss_test', action='store_true', default=False) #To ablate without alignment loss

    parser.add_argument('--use_cosine_aux_loss', action='store_true', default=False)
    parser.add_argument('--aux_loss_type', default='cosine', choices=['cosine', 'contrastive-InfoNCE', 'constrastive-margin'])
    parser.add_argument("--cosine_weight", type=float, default=0.20)
    parser.add_argument("--infonce_temperature", type=float, default=0.30)
    parser.add_argument("--contrastive_margin_value", type=float, default=0.5)

    parser.add_argument('--concat_imagine_with', default='visual', choices=['visual','language'])

    parser.add_argument("--imag_train_features", type=str, default='vitbase_imagination_train')
    parser.add_argument("--imag_val_seen_features", type=str, default='vitbase_imagination_val_seen')
    parser.add_argument("--imag_val_unseen_features", type=str, default='vitbase_imagination_val_unseen')
    parser.add_argument("--imag_test_features", type=str, default='vitbase_imagination_test')

    parser.add_argument('--imagine_features_type', default='hamt-vit', choices=['hamt-vit','off-the-shelf-vit','off-the-shelf-vit-clip'])
    
    args, _ = parser.parse_known_args()

    args = postprocess_args(args)

    return args


def postprocess_args(args):
    ROOTDIR = args.root_dir

    # Setup input paths
    ft_file_map = {
        'vitbase': 'pth_vit_base_patch16_224_imagenet.hdf5',
        'vitbase_r2rfte2e': 'pth_vit_base_patch16_224_imagenet_r2r.e2e.ft.22k.hdf5',
        'vitbase_clip': 'pth_vit_base_patch32_224_clip.hdf5',
        'vitbase_imagination_train': 'pth_vit_base_patch16_224_imagenet_imagine_train.hdf5',
        'vitbase_imagination_val_seen': 'pth_vit_base_patch16_224_imagenet_imagine_val_seen.hdf5',
        'vitbase_imagination_val_unseen': 'pth_vit_base_patch16_224_imagenet_imagine_val_unseen.hdf5',
        'vitbase_imagination_test': 'pth_vit_base_patch16_224_imagenet_imagine_test.hdf5'
    }
    args.img_ft_file = os.path.join(ROOTDIR, 'R2R', 'features', ft_file_map[args.features])
    
    #If using new imagination data (v2) which may not have imaginations for any sub-instr in a particular instr_idx. This flag is to allow reading the generated json file which has generated flag.
    if args.imagination_data_v2:
        imagine_train_v2_json = os.path.join('/nfs/hpc/sw_data/perincha/r2r_imaginations-v2/train', '2024_05_08-12_22_38_AM_results.json')
        with open(imagine_train_v2_json, "r") as f: 
            imagine_train_v2_data = json.loads(f.read())
        args.imagine_train_v2_generated_flag = {str(d['path_id'])+'_'+str(d['instruction']):d['generated_imaginations'] for d in imagine_train_v2_data}

        imagine_val_seen_v2_json = os.path.join('/nfs/hpc/sw_data/perincha/r2r_imaginations-v2/val-seen', '2024_05_03-09_44_47_PM_results.json')
        with open(imagine_val_seen_v2_json, "r") as f: 
            imagine_val_seen_v2_data = json.loads(f.read())
        args.imagine_val_seen_v2_generated_flag = {str(d['path_id'])+'_'+str(d['instruction']):d['generated_imaginations'] for d in imagine_val_seen_v2_data}

        imagine_val_unseen_v2_json = os.path.join('/nfs/hpc/sw_data/perincha/r2r_imaginations-v2/val-unseen', '2024_05_03-09_52_58_PM_results.json')
        with open(imagine_val_unseen_v2_json, "r") as f: 
            imagine_val_unseen_v2_data = json.loads(f.read())
        args.imagine_val_unseen_v2_generated_flag = {str(d['path_id'])+'_'+str(d['instruction']):d['generated_imaginations'] for d in imagine_val_unseen_v2_data}

        imagine_test_v2_json = os.path.join('/nfs/hpc/sw_data/perincha/r2r_imaginations-v2/test', '2024_05_04-10_52_11_AM_results.json')
        with open(imagine_test_v2_json, "r") as f: 
            imagine_test_v2_data = json.loads(f.read())
        args.imagine_test_v2_generated_flag = {str(d['path_id'])+'_'+str(d['instruction']):d['generated_imaginations'] for d in imagine_test_v2_data}

        if args.imagine_features_type == 'off-the-shelf-vit':
            print('Reading off-the-shelf v2 imagination features.')
            args.imagination_train_ft_file = os.path.join(ROOTDIR, 'R2R', 'features', 'vit-off-the-shelf-imagination-features-v2', ft_file_map[args.imag_train_features])
            args.imagination_val_seen_ft_file = os.path.join(ROOTDIR, 'R2R', 'features', 'vit-off-the-shelf-imagination-features-v2', ft_file_map[args.imag_val_seen_features])
            args.imagination_val_unseen_ft_file = os.path.join(ROOTDIR, 'R2R', 'features', 'vit-off-the-shelf-imagination-features-v2', ft_file_map[args.imag_val_unseen_features])
            args.imagination_test_ft_file = os.path.join(ROOTDIR, 'R2R', 'features', 'vit-off-the-shelf-imagination-features-v2', ft_file_map[args.imag_test_features])
        
        elif args.imagine_features_type == 'hamt-vit':
            print('Reading vit-hamt v2 imagination features.')
            args.imagination_train_ft_file = os.path.join(ROOTDIR, 'R2R', 'features', 'vit-hamt-imagination-features-v2', ft_file_map[args.imag_train_features])
            args.imagination_val_seen_ft_file = os.path.join(ROOTDIR, 'R2R', 'features', 'vit-hamt-imagination-features-v2', ft_file_map[args.imag_val_seen_features])
            args.imagination_val_unseen_ft_file = os.path.join(ROOTDIR, 'R2R', 'features', 'vit-hamt-imagination-features-v2', ft_file_map[args.imag_val_unseen_features])
            args.imagination_test_ft_file = os.path.join(ROOTDIR, 'R2R', 'features', 'vit-hamt-imagination-features-v2', ft_file_map[args.imag_test_features])
        
        else:
            raise ValueError('Invalid imagine_features_type: '+args.imagine_features_type)
            
    args.connectivity_dir = os.path.join(ROOTDIR, 'R2R', 'connectivity')
    args.scan_data_dir = os.path.join(ROOTDIR, 'Matterport3D', 'v1_unzip_scans')

    if args.dataset == 'rxr':
        args.anno_dir = os.path.join(ROOTDIR, 'RxR', 'annotations')
    else:
        args.anno_dir = os.path.join(ROOTDIR, 'R2R', 'annotations')
        args.anno_dir_fg_r2r = os.path.join(args.root_dir_fgr2r, 'data')
    
    args.sub_instr_seg_dir = os.path.join(ROOTDIR, 'R2R', 'annotations', 'FGR2R_nounphrase_segmentations') #This contains sub-instr and noun-phrase information.

    # Build paths
    args.ckpt_dir = os.path.join(args.output_dir, 'ckpts')
    args.log_dir = os.path.join(args.output_dir, 'logs')
    args.pred_dir = os.path.join(args.output_dir, 'preds')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.pred_dir, exist_ok=True)

    # remove unnecessary args
    if args.dataset != 'rxr':
        del args.langs

    return args

