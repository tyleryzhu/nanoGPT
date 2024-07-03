torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py \
    --out_dir="out-rev-vanilla-gpt2-bs32" \
    --wandb_run_name="rev-vanilla-gpt2-bs32" \
    --gradient_accumulation_steps=16 \
    --batch_size=32 \
    --reversible=True \
    --vanilla_backward=True \
    --wandb_log=True
