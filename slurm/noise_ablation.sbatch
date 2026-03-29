#!/bin/bash
#SBATCH --job-name=noise-abl
#SBATCH --partition=gpu-H200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-noise-abl-%j.out
#SBATCH --error=slurm-noise-abl-%j.err

module load python3/anaconda/3.12

echo "Noise ablation at $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

cd $HOME/edge-detection-filter-critique
pip install --user -e . 2>&1 | tail -1

python3 -u scripts/run_noise_ablation.py
