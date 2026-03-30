#!/bin/bash
#SBATCH --job-name=dl-diffusion
#SBATCH --partition=gpu-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-dl-diffusion-%j.out
#SBATCH --error=slurm-dl-diffusion-%j.err

# DiffusionEdge needs more memory and time than the others

module load python3/anaconda/3.12

echo "DiffusionEdge test at $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

cd $HOME/edge-detection-filter-critique
pip install --user -e . 2>&1 | tail -1
pip install --user omegaconf ema-pytorch albumentations fvcore 2>&1 | tail -1

python3 -u scripts/run_dl_single_image.py --model diffusionedge
