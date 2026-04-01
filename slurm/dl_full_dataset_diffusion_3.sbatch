#!/bin/bash
#SBATCH --job-name=dl-diff-3
#SBATCH --partition=gpu-H200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-dl-diff-3-%j.out
#SBATCH --error=slurm-dl-diff-3-%j.err

# DiffusionEdge full-dataset ODS: BSDS500 (200 imgs) — largest dataset

module load python3/anaconda/3.12

echo "DiffusionEdge full-dataset ODS (BSDS500) at $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

cd $HOME/edge-detection-filter-critique
pip install --user -e . 2>&1 | tail -1
pip install --user omegaconf ema-pytorch albumentations fvcore 2>&1 | tail -1

python3 -u scripts/run_full_dataset_dl.py --model diffusionedge --datasets bsds500
