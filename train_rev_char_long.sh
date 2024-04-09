out_dir=out-rev-char-long
mkdir $out_dir

python train.py config/train_shakespeare_char.py \
    --device=mps \
    --compile=False \
    --eval_iters=20 \
    --eval_interval=250 \
    --log_interval=1 \
    --block_size=64 \
    --batch_size=12 \
    --n_layer=24 \
    --n_head=4 \
    --n_embd=64 \
    --max_iters=2000 \
    --lr_decay_iters=2000 \
    --dropout=0.0 \
    --reversible=True \
    --out_dir=$out_dir \
    > "${out_dir}/out.log"