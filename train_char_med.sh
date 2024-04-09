out_dir=out-char-med

python train.py config/train_shakespeare_char.py \
    --device=mps \
    --compile=False \
    --eval_iters=500 \
    --eval_interval=250 \
    --log_interval=1 \
    --block_size=64 \
    --batch_size=32 \
    --n_layer=8 \
    --n_head=16 \
    --n_embd=512 \
    --max_iters=2000 \
    --lr_decay_iters=2000 \
    --dropout=0.0 \
    --out_dir=$out_dir \
    > "${out_dir}/char.log"