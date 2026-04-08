#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=visualai
#SBATCH -o slurm/validate_%j.out
#SBATCH -e slurm/validate_%j.err
#SBATCH -t 0-1:00:00

source ~/.bashrc
conda activate home
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd /n/fs/tz-ego4d/nanoGPT

echo "========== GRADIENT MATCHING TEST =========="
python test_gradient_match.py

echo ""
echo "========== TRAINING: standard char-tiny (500 iters) =========="
out_dir=out-validate-char
mkdir -p $out_dir
python train.py config/train_shakespeare_char.py \
    --device=cuda \
    --compile=False \
    --eval_iters=20 \
    --eval_interval=100 \
    --log_interval=50 \
    --block_size=64 \
    --batch_size=32 \
    --n_layer=4 \
    --n_head=4 \
    --n_embd=128 \
    --max_iters=500 \
    --lr_decay_iters=500 \
    --dropout=0.0 \
    --out_dir=$out_dir

echo ""
echo "========== TRAINING: reversible char-tiny (500 iters) =========="
out_dir=out-validate-rev-char
mkdir -p $out_dir
python train.py config/train_shakespeare_char.py \
    --device=cuda \
    --compile=False \
    --eval_iters=20 \
    --eval_interval=100 \
    --log_interval=50 \
    --block_size=64 \
    --batch_size=32 \
    --n_layer=4 \
    --n_head=4 \
    --n_embd=128 \
    --max_iters=500 \
    --lr_decay_iters=500 \
    --dropout=0.0 \
    --reversible=True \
    --out_dir=$out_dir

echo ""
echo "========== TRAINING: reversible vanilla-backward char-tiny (500 iters) =========="
out_dir=out-validate-rev-vanilla-char
mkdir -p $out_dir
python train.py config/train_shakespeare_char.py \
    --device=cuda \
    --compile=False \
    --eval_iters=20 \
    --eval_interval=100 \
    --log_interval=50 \
    --block_size=64 \
    --batch_size=32 \
    --n_layer=4 \
    --n_head=4 \
    --n_embd=128 \
    --max_iters=500 \
    --lr_decay_iters=500 \
    --dropout=0.0 \
    --reversible=True \
    --vanilla_backward=True \
    --out_dir=$out_dir
