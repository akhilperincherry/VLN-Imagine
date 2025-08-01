DATA_ROOT=../datasets

IMAGINATION_ROOT_FOLDER=/nfs/hpc/sw/perincha/repos/VLN-DUET/datasets/REVERIE/imagination_experiments

train_alg=dagger

features=vitbase

imag_train_features=vitbase_imagination_train
imag_val_seen_features=vitbase_imagination_val_seen
imag_val_unseen_features=vitbase_imagination_val_unseen
imag_test_features=vitbase_imagination_test

ft_dim=768
obj_features=vitbase
obj_ft_dim=768

ngpus=1
seed=0

curr_imagination_exp=imagination-reverie-cosine05

name=${train_alg}-${features}-${curr_imagination_exp}
# name=${name}-seed.${seed} #-${ngpus}gpus

outdir=${IMAGINATION_ROOT_FOLDER}/experiment_folder/${name}

flag="--root_dir ${DATA_ROOT}
      --dataset reverie
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert

      --enc_full_graph
      --graph_sprels
      --fusion dynamic
      --multi_endpoints

      --dagger_sample sample

      --train_alg ${train_alg}
      
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      
      --max_action_len 15
      --max_instr_len 200
      --max_objects 20

      --batch_size 8
      --lr 1e-5
      --iters 100000
      --log_every 2000
      --optim adamW

      --features ${features}
      --obj_features ${obj_features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4
      --obj_feat_size ${obj_ft_dim}

      --imag_train_features ${imag_train_features}
      --imag_val_seen_features ${imag_val_seen_features}
      --imag_val_unseen_features ${imag_val_unseen_features}
      --imag_test_features ${imag_test_features}

      --ml_weight 0.2

      --feat_dropout 0.4
      --dropout 0.5
      
      --gamma 0.
      
      --imagine_enc_pano
      --imagination_data_v2
      --max_imagination_len 20
      --bypass_imag_encoder
      --use_cosine_aux_loss
      --experimental_warmup
      --cosine_weight 0.5
      --aux_loss_type cosine
      --concat_imagine_with language
      --imagine_features_type off-the-shelf-vit
      --experimental_warmup_type variant4"
      
# train
# CUDA_VISIBLE_DEVICES=0 python reverie/main_nav_obj.py $flag --resume_file ../datasets/REVERIE/trained_models/best_val_unseen --eval_first

# test
CUDA_VISIBLE_DEVICES='0' python reverie/main_nav_obj.py $flag  \
      --tokenizer bert \
      --resume_file <ckpt> \
      --test
 