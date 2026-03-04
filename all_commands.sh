conda activate malinois_inference
module use /projects/community/modulefiles
module load git


python train.py \
    --data_csv data/MPAC_emvar_K562_combined.tsv \
    --ref_col ref_seq --alt_col alt_seq --ref_act_col ref_log2FC --alt_act_col alt_log2FC \
    --pretrained model/sei.pth \
    --epochs 20 \
    --hidden_dim 2048 \
    --optimizer adam \
    --lr_head 1e-3 \
    --lr_backbone 1e-5 \
    --batch_size 32 \
    --freeze_backbone


python predict_sei.py \
  --input_csv data/example_sequences.csv \
  --pretrained model/sei.pth \
  --batch_size 128 \
  --out_npy data/sei_preds.npy