#!/bin/bash
#SBATCH --job-name=dl-fast
#SBATCH --partition=gpu-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-dl-fast-%j.out
#SBATCH --error=slurm-dl-fast-%j.err

# Quick test: run one or all DL models on a single BIPED image
# Usage: sbatch slurm/dl_test_fast.sbatch           (runs all)
#        sbatch slurm/dl_test_fast.sbatch dexined    (runs one)

MODEL=${1:-all}

module load python3/anaconda/3.12

echo "DL model test ($MODEL) at $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

cd $HOME/edge-detection-filter-critique
pip install --user -e . 2>&1 | tail -1
pip install --user kornia "timm==0.6.13" albumentations 2>&1 | tail -1

python3 -u scripts/run_dl_single_image.py --model $MODEL
