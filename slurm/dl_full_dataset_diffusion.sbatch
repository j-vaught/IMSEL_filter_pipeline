#!/bin/bash
#SBATCH --job-name=dl-diff-all
#SBATCH --partition=gpu-H200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-dl-diff-all-%j.out
#SBATCH --error=slurm-dl-diff-all-%j.err

# DiffusionEdge full-dataset ODS: all 4 datasets sequentially.
# Single job to avoid race condition on shared tmp dirs.

module load python3/anaconda/3.12

echo "DiffusionEdge full-dataset ODS (all datasets) at $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

cd $HOME/edge-detection-filter-critique
pip install --user -e . 2>&1 | tail -1
pip install --user omegaconf ema-pytorch albumentations fvcore 2>&1 | tail -1

python3 -u scripts/run_full_dataset_dl.py --model diffusionedge
