torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py \
    --out_dir="out-rev-gpt2-bs32-autocastfix" \
    --wandb_run_name="rev-gpt2-bs32-autocastfix" \
    --gradient_accumulation_steps=16 \
    --batch_size=32 \
    --reversible=True \
    --vanilla_backward=False \
    --wandb_log=True
