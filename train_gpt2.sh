torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py \
    --out_dir="out-gpt2-bs32" \
    --wandb_run_name="gpt2-bs32" \
    --gradient_accumulation_steps=8 \
    --batch_size=32 \
    --wandb_log=True