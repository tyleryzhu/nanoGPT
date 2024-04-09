out_dir=out-rev-char-tiny
mkdir $out_dir

python train.py config/train_shakespeare_char.py \
    --device=mps \
    --compile=False \
    --eval_iters=200 \
    --eval_interval=100 \
    --log_interval=1 \
    --block_size=64 \
    --batch_size=32 \
    --n_layer=4 \
    --n_head=4 \
    --n_embd=128 \
    --max_iters=2500 \
    --lr_decay_iters=2500 \
    --dropout=0.0 \
    --reversible=True \
    --out_dir=$out_dir \
    > "${out_dir}/out.log"