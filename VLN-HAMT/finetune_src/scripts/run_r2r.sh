ob_type=pano
feedback=sample

features=vitbase_r2rfte2e
# features=vitbase

imag_train_features=vitbase_imagination_train
imag_val_seen_features=vitbase_imagination_val_seen
imag_val_unseen_features=vitbase_imagination_val_unseen
imag_test_features=vitbase_imagination_test
ft_dim=768

ngpus=1
seed=0

outdir=imagination-exp-1

flag="--root_dir ../datasets
      --output_dir ${outdir}

      --dataset r2r

      --vlnbert ${vlnbert}
      --ob_type ${ob_type}
      
      --world_size ${ngpus}
      --seed ${seed}
      
      --num_l_layers 9
      --num_x_layers 4
      
      --hist_enc_pano
      --hist_pano_num_layers 2

      --fix_lang_embedding
      --fix_hist_embedding

      --features ${features}
      
      --imag_train_features ${imag_train_features}
      --imag_val_seen_features ${imag_val_seen_features}
      --imag_val_unseen_features ${imag_val_unseen_features}
      --imag_test_features ${imag_test_features}

      --feedback ${feedback}

      --max_action_len 15
      --max_instr_len 60

      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --lr 1e-5
      --iters 100000
      --log_every 2000
      --batch_size 8
      --optim adamW

      --ml_weight 0.2      

      --feat_dropout 0.4
      --dropout 0.5
      
      --imagine_enc_pano
      --imagination_data_v2
      --act_pred_token ob_txt
      --max_imagination_len 20
      --wrong_diffusion_inference_setting na
      --null_diffusion_inference_setting na
      --bypass_imag_encoder
      --use_cosine_aux_loss
      --experimental_warmup
      --cosine_weight 0.5
      --concat_imagine_with language
      --imagine_features_type off-the-shelf-vit
      --experimental_warmup_type variant4" 

# train
# CUDA_VISIBLE_DEVICES=0 python r2r/main.py $flag --resume_file ../datasets/R2R/trained_models/vitbase-finetune-e2e/ckpts/best_val_unseen

# inference
CUDA_VISIBLE_DEVICES=0 python r2r/main.py $flag \
--resume_file iter_32000_SR_67.260000_SPL_62.020000_val_unseen --test