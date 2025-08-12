DATA_ROOT=../datasets

IMAGINATION_ROOT_FOLDER=imagination-experiments

train_alg=dagger

features=vitbase
ft_dim=768
obj_features=vitbase
obj_ft_dim=768

imag_train_features=vitbase_imagination_train
imag_val_seen_features=vitbase_imagination_val_seen
imag_val_unseen_features=vitbase_imagination_val_unseen
imag_test_features=vitbase_imagination_test

ngpus=1
seed=0

curr_imagination_exp=imagination-exp-1

name=${train_alg}-${features}-${curr_imagination_exp}
name=${name}-seed.${seed}
# name=${name}-init.aug.45k

outdir=${IMAGINATION_ROOT_FOLDER}/experimental-runs/${name}

flag="--root_dir ${DATA_ROOT}
      --dataset r2r
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert      

      --enc_full_graph
      --graph_sprels
      --fusion dynamic

      --expert_policy spl
      --train_alg ${train_alg}
      
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      
      --max_action_len 15
      --max_instr_len 200

      --batch_size 8
      --lr 1e-5
      --iters 100000
      --log_every 2000
      --optim adamW

      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --imag_train_features ${imag_train_features}
      --imag_val_seen_features ${imag_val_seen_features}
      --imag_val_unseen_features ${imag_val_unseen_features}
      --imag_test_features ${imag_test_features}

      --ml_weight 0.2   

      --feat_dropout 0.4
      --dropout 0.5
      
      --gamma 0.
      
      --fix_lang_inside_cosine_model
      --imagine_enc_pano
      --imagination_data_v2
      --max_imagination_len 20
      --bypass_imag_encoder
      --use_cosine_aux_loss
      --experimental_warmup
      --cosine_weight 0.5
      --concat_imagine_with language
      --imagine_features_type off-the-shelf-vit
      --experimental_warmup_type variant4"

# test
CUDA_VISIBLE_DEVICES='0' python r2r/main_nav.py $flag  \
      --tokenizer bert \
      --resume_file iter_44000_SR_72.120000_SPL_60.480000_val_unseen \
      --test

# train
# CUDA_VISIBLE_DEVICES='0' python r2r/main_nav.py $flag  \
#       --tokenizer bert \
#       --resume_file ../datasets/R2R/trained_models/best_val_unseen \
#       --eval_first