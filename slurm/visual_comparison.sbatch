#!/bin/bash
#SBATCH --job-name=asf-visuals
#SBATCH --partition=gpu-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-asf-visuals-%j.out
#SBATCH --error=slurm-asf-visuals-%j.err

module load python3/anaconda/3.12

echo "ASF Visual Comparison at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd $HOME/edge-detection-filter-critique
pip install --user -e . 2>&1 | tail -1
pip install --user triton 2>&1 | tail -1

python3 -u scripts/generate_visual_comparison.py

echo "Done at $(date)"
