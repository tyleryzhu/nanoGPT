#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=visualai
#SBATCH -o slurm/diagnose_%j.out
#SBATCH -e slurm/diagnose_%j.err
#SBATCH -t 0-0:30:00

source ~/.bashrc
conda activate home
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd /n/fs/tz-ego4d/nanoGPT

echo "========== DIAGNOSTIC SCRIPT =========="
python diagnose_reversibility.py

echo ""
echo "========== EXISTING BASELINE TEST (reversible_model.py __main__) =========="
python reversible_model.py
