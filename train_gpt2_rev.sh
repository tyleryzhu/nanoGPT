torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py \
    --out_dir="out-rev-gpt2" \
    --wandb_run_name="rev-gpt2" \
    --reversible=True \
    --wandb_log=True