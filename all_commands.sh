srun --partition=gpu --gres=gpu:4 --pty --mem=48G -t 120:00 bash
conda activate malinois_inference
module use /projects/community/modulefiles
module load git

python train.py \
    --data_csv data/MPAC_emvar_K562_combined.tsv \
    --ref_col ref_seq --alt_col alt_seq --ref_act_col ref_log2FC --alt_act_col alt_log2FC \
    --pretrained model/sei.pth \
    --epochs 5 \
    --hidden_dim 2048 \
    --optimizer adam \
    --lr_head 3e-5 \
    --lr_backbone 1e-5 \
    --batch_size 128 \
    --freeze_backbone

# gpu016 -- running quite fast here (20/22 sec per epoch)
# 1e-4 with 256 -- 0.59
# 1e-4 with 512 -- 


python train.py \
    --data_csv data/MPAC_emvar_K562_combined.tsv \
    --ref_col ref_seq --alt_col alt_seq --ref_act_col ref_log2FC --alt_act_col alt_log2FC \
    --pretrained model/sei.pth \
    --epochs 5 \
    --hidden_dim 4096 \
    --optimizer adam \
    --lr_head 1e-5 \
    --lr_backbone 1e-5 \
    --batch_size 256 \
    --freeze_backbone


python predict_sei.py \
  --input_csv data/example_sequences.csv \
  --pretrained model/sei.pth \
  --batch_size 128 \
  --out_npy data/sei_preds.npy


python train_w_optuna.py \
  --data_csv data/MPAC_emvar_K562_combined.tsv \
  --ref_col ref_seq --alt_col alt_seq --ref_act_col ref_log2FC --alt_act_col alt_log2FC \
  --pretrained model/sei.pth \
  --epochs 10 \
  --n_trials 30 \
  --search_batch_size \
  --search_hidden_dim \
  --search_lr_head \
  --search_weight_decay \
  --search_optimizer \
  --use_pruner \
  --storage sqlite:///optuna_study.db \
  --freeze_backbone