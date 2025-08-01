import argparse
import os
import json
import jsonlines

def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--root_dir', type=str, default='../datasets')
    parser.add_argument('--dataset', type=str, default='reverie', choices=['reverie'])
    parser.add_argument('--output_dir', type=str, default='default', help='experiment id')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--tokenizer', choices=['bert', 'xlm'], default='bert')

    parser.add_argument('--fusion', choices=['global', 'local', 'avg', 'dynamic'])
    parser.add_argument('--dagger_sample', choices=['sample', 'expl_sample', 'argmax'])
    parser.add_argument('--expl_max_ratio', type=float, default=0.6)
    parser.add_argument('--loss_nav_3', action='store_true', default=False)

    # distributional training (single-node, multiple-gpus)
    parser.add_argument('--world_size', type=int, default=1, help='number of gpus')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument("--node_rank", type=int, default=0, help="Id of the node")
    
    # General
    parser.add_argument('--iters', type=int, default=100000, help='training iterations')
    parser.add_argument('--log_every', type=int, default=1000)
    parser.add_argument('--eval_first', action='store_true', default=False)

    # Data preparation
    parser.add_argument('--max_instr_len', type=int, default=80)
    parser.add_argument('--max_action_len', type=int, default=15)
    parser.add_argument('--max_objects', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--ignoreid', type=int, default=-100, help='ignoreid for action')
    
    # Load the model from
    parser.add_argument("--resume_file", default=None, help='path of the trained model')
    parser.add_argument("--resume_optimizer", action="store_true", default=False)

    # Augmented Paths from
    parser.add_argument("--multi_endpoints", default=False, action="store_true")
    parser.add_argument("--multi_startpoints", default=False, action="store_true")
    parser.add_argument("--aug_only", default=False, action="store_true")
    parser.add_argument("--aug", default=None)
    parser.add_argument('--bert_ckpt_file', default=None, help='init vlnbert')

    # Listener Model Config
    parser.add_argument("--ml_weight", type=float, default=0.20)
    parser.add_argument('--entropy_loss_weight', type=float, default=0.01)

    parser.add_argument("--features", type=str, default='vitbase')
    parser.add_argument('--obj_features', type=str, default='vitbase')

    parser.add_argument('--fix_lang_embedding', action='store_true', default=False)
    parser.add_argument('--fix_pano_embedding', action='store_true', default=False)
    parser.add_argument('--fix_local_branch', action='store_true', default=False)

    parser.add_argument('--num_l_layers', type=int, default=9)
    parser.add_argument('--num_pano_layers', type=int, default=2)
    parser.add_argument('--num_x_layers', type=int, default=4)

    parser.add_argument('--enc_full_graph', default=False, action='store_true')
    parser.add_argument('--graph_sprels', action='store_true', default=False)

    # Dropout Param
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--feat_dropout', type=float, default=0.3)

    # Submision configuration
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument("--submit", action='store_true', default=False)
    parser.add_argument('--no_backtrack', action='store_true', default=False)
    parser.add_argument('--detailed_output', action='store_true', default=False)

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
    parser.add_argument('--epsilon', type=float, default=0.1, help='')

    # Model hyper params:
    parser.add_argument("--angle_feat_size", type=int, default=4)
    parser.add_argument('--image_feat_size', type=int, default=2048)
    parser.add_argument('--obj_feat_size', type=int, default=2048)
    parser.add_argument('--views', type=int, default=36)

    # # A2C
    parser.add_argument("--gamma", default=0.9, type=float, help='reward discount factor')
    parser.add_argument(
        "--normalize", dest="normalize_loss", default="total", 
        type=str, help='batch or total'
    )
    parser.add_argument('--train_alg', 
        choices=['imitation', 'dagger'], 
        default='imitation'
    )

    parser.add_argument('--imagine_enc_pano', action='store_true', default=False)
    parser.add_argument('--imagination_data_v2', action='store_true', default=False)
    parser.add_argument('--bypass_imag_encoder', action='store_true', default=False) #Pass imag features directly to CMT.
    parser.add_argument('--fix_imagine_embeds', action='store_true', default=False)
    parser.add_argument('--max_imagination_len', type=int, default=25)
    parser.add_argument('--experimental_warmup', action='store_true', default=False)
    parser.add_argument('--experimental_warmup_type', default='variant4', choices=['variant4'])
    parser.add_argument('--no_loss_test', action='store_true', default=False) #To ablate without alignment loss
    parser.add_argument('--use_cosine_aux_loss', action='store_true', default=False)
    parser.add_argument('--aux_loss_type', default='cosine', choices=['cosine', 'contrastive-InfoNCE', 'constrastive-margin'])
    parser.add_argument("--cosine_weight", type=float, default=0.20)
    parser.add_argument("--infonce_temperature", type=float, default=0.007)
    parser.add_argument("--contrastive_margin_value", type=float, default=0.5)

    parser.add_argument('--concat_imagine_with', default='language', choices=['visual','language'])

    parser.add_argument("--imag_train_features", type=str, default='vitbase_imagination_train')
    parser.add_argument("--imag_val_seen_features", type=str, default='vitbase_imagination_val_seen')
    parser.add_argument("--imag_val_unseen_features", type=str, default='vitbase_imagination_val_unseen')
    parser.add_argument("--imag_test_features", type=str, default='vitbase_imagination_test')

    parser.add_argument('--imagine_features_type', default='off-the-shelf-vit', choices=['hamt-vit','off-the-shelf-vit', 'off-the-shelf-vit-clip'])

    parser.add_argument('--fix_lang_inside_cosine_model', action='store_true', default=False)
    parser.add_argument('--use_dropout_on_imagine', action='store_true', default=False)

    args, _ = parser.parse_known_args()

    args = postprocess_args(args)

    return args


def postprocess_args(args):
    ROOTDIR = args.root_dir
    # path to root folder containing imagination features.
    HAMT_ROOTDIR='/nfs/hpc/sw/perincha/repos/VLN-HAMT/datasets'

    # Setup input paths
    ft_file_map = {
        'vitbase': 'pth_vit_base_patch16_224_imagenet.hdf5',
    }
    args.img_ft_file = os.path.join(ROOTDIR, 'R2R', 'features', ft_file_map[args.features])

    obj_ft_file_map = {
        'vitbase': 'obj.avg.top3.min80_vit_base_patch16_224_imagenet.hdf5',
    }
    args.obj_ft_file = os.path.join(ROOTDIR, 'REVERIE', 'features', obj_ft_file_map[args.obj_features])


    imagine_ft_file_map = {
        'vitbase_imagination_train': 'pth_vit_base_patch16_224_imagenet_imagine_train.hdf5',
        'vitbase_imagination_val_seen': 'pth_vit_base_patch16_224_imagenet_imagine_val_seen.hdf5',
        'vitbase_imagination_val_unseen': 'pth_vit_base_patch16_224_imagenet_imagine_val_unseen.hdf5',
        'vitbase_imagination_test': 'pth_vit_base_patch16_224_imagenet_imagine_test.hdf5'
    }

    if args.imagine_features_type == 'off-the-shelf-vit':
        print('Reading off-the-shelf v2 imagination features.')
        args.imagination_train_ft_file = os.path.join(HAMT_ROOTDIR, 'REVERIE', 'features', 'vit-off-the-shelf-imagination-features-v2', imagine_ft_file_map[args.imag_train_features])
        args.imagination_val_seen_ft_file = os.path.join(HAMT_ROOTDIR, 'REVERIE', 'features', 'vit-off-the-shelf-imagination-features-v2', imagine_ft_file_map[args.imag_val_seen_features])
        args.imagination_val_unseen_ft_file = os.path.join(HAMT_ROOTDIR, 'REVERIE', 'features', 'vit-off-the-shelf-imagination-features-v2', imagine_ft_file_map[args.imag_val_unseen_features])
        args.imagination_test_ft_file = os.path.join(HAMT_ROOTDIR, 'REVERIE', 'features', 'vit-off-the-shelf-imagination-features-v2', imagine_ft_file_map[args.imag_test_features])

    elif args.imagine_features_type == 'off-the-shelf-vit-clip':
        print('Reading off-the-shelf v2 CLIP imagination features.')
        args.imagination_train_ft_file = os.path.join(HAMT_ROOTDIR, 'REVERIE', 'features', 'vit-off-the-shelf-imagination-features-v2', 'clip', imagine_ft_file_map[args.imag_train_features])
        args.imagination_val_seen_ft_file = os.path.join(HAMT_ROOTDIR, 'REVERIE', 'features', 'vit-off-the-shelf-imagination-features-v2', 'clip', imagine_ft_file_map[args.imag_val_seen_features])
        args.imagination_val_unseen_ft_file = os.path.join(HAMT_ROOTDIR, 'REVERIE', 'features', 'vit-off-the-shelf-imagination-features-v2', 'clip', imagine_ft_file_map[args.imag_val_unseen_features])
        args.imagination_test_ft_file = os.path.join(HAMT_ROOTDIR, 'REVERIE', 'features', 'vit-off-the-shelf-imagination-features-v2', 'clip', imagine_ft_file_map[args.imag_test_features])

    elif args.imagine_features_type == 'hamt-vit':
        print('Reading vit-hamt v2 imagination features.')
        args.imagination_train_ft_file = os.path.join(HAMT_ROOTDIR, 'REVERIE', 'features', 'vit-hamt-imagination-features-v2', imagine_ft_file_map[args.imag_train_features])
        args.imagination_val_seen_ft_file = os.path.join(HAMT_ROOTDIR, 'REVERIE', 'features', 'vit-hamt-imagination-features-v2', imagine_ft_file_map[args.imag_val_seen_features])
        args.imagination_val_unseen_ft_file = os.path.join(HAMT_ROOTDIR, 'REVERIE', 'features', 'vit-hamt-imagination-features-v2', imagine_ft_file_map[args.imag_val_unseen_features])
        args.imagination_test_ft_file = os.path.join(HAMT_ROOTDIR, 'REVERIE', 'features', 'vit-hamt-imagination-features-v2', imagine_ft_file_map[args.imag_test_features])
    
    else:
        raise ValueError('Invalid imagine_features_type: '+args.imagine_features_type)
    
    args.connectivity_dir = os.path.join(ROOTDIR, 'R2R', 'connectivity')
    args.scan_data_dir = os.path.join(ROOTDIR, 'Matterport3D', 'v1_unzip_scans')

    args.anno_dir = os.path.join(ROOTDIR, 'REVERIE', 'annotations')

    # Build paths
    args.ckpt_dir = os.path.join(args.output_dir, 'ckpts')
    args.log_dir = os.path.join(args.output_dir, 'logs')
    args.pred_dir = os.path.join(args.output_dir, 'preds')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.pred_dir, exist_ok=True)

    return args

