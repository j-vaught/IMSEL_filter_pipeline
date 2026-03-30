#!/bin/bash
#SBATCH --job-name=dl-nbed
#SBATCH --partition=gpu-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-dl-nbed-%j.out
#SBATCH --error=slurm-dl-nbed-%j.err

module load python3/anaconda/3.12

echo "NBED noise ablation at $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

cd $HOME/edge-detection-filter-critique
pip install --user -e . 2>&1 | tail -1
pip install --user "timm==0.6.13" 2>&1 | tail -1

python3 -u scripts/run_dl_noise_ablation.py --model nbed
