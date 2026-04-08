torchrun --standalone --nproc_per_node=1 bench.py \
    --batch_size=16 \
    --profile=True \
    # --reversible=True
    # --gradient_accumulation_steps=8 \