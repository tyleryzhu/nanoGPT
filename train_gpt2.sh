torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py \
    --out_dir="out-gpt2" \
    --wandb_run_name="gpt2" \
    --wandb_log=True