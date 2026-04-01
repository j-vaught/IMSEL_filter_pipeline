#!/bin/bash
#SBATCH --job-name=dl-diff-1
#SBATCH --partition=gpu-H200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-dl-diff-1-%j.out
#SBATCH --error=slurm-dl-diff-1-%j.err

# DiffusionEdge full-dataset ODS: UDED (30 imgs) + BIPED_v1 (50 imgs)

module load python3/anaconda/3.12

echo "DiffusionEdge full-dataset ODS (UDED + BIPED_v1) at $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

cd $HOME/edge-detection-filter-critique
pip install --user -e . 2>&1 | tail -1
pip install --user omegaconf ema-pytorch albumentations fvcore 2>&1 | tail -1

python3 -u scripts/run_full_dataset_dl.py --model diffusionedge --datasets uded biped_v1
