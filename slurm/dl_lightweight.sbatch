#!/bin/bash
#SBATCH --job-name=dl-light
#SBATCH --partition=gpu-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-dl-light-%j.out
#SBATCH --error=slurm-dl-light-%j.err

module load python3/anaconda/3.12

echo "DexiNed + TEED + pidinet noise ablation at $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

cd $HOME/edge-detection-filter-critique
pip install --user -e . 2>&1 | tail -1
pip install --user kornia "timm==0.6.13" 2>&1 | tail -1

for MODEL in dexined teed pidinet; do
    echo "########## $MODEL ##########"
    python3 -u scripts/run_dl_noise_ablation.py --model $MODEL
    echo
done
