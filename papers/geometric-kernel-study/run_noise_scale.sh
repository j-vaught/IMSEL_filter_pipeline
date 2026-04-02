#!/bin/bash
#SBATCH --job-name=filter-noise-scale
#SBATCH --partition=gpu-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/jvaught/testing/BSDS500/noise_scale_%j.out
#SBATCH --error=/home/jvaught/testing/BSDS500/noise_scale_%j.err

cd /home/jvaught/testing

module load python 2>/dev/null || true
module load cuda 2>/dev/null || true

python noise_vs_scale_gpu.py
