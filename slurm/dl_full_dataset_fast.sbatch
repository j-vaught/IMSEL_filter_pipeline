#!/bin/bash
#SBATCH --job-name=dl-full-fast
#SBATCH --partition=gpu-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-dl-full-fast-%j.out
#SBATCH --error=slurm-dl-full-fast-%j.err

# Full-dataset ODS for the 4 lightweight DL models (~7 min inference + eval)

module load python3/anaconda/3.12

echo "Full-dataset DL ODS (fast models) at $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

cd $HOME/edge-detection-filter-critique
pip install --user -e . 2>&1 | tail -1
pip install --user kornia "timm==0.6.13" albumentations 2>&1 | tail -1

for MODEL in pidinet dexined teed nbed; do
    echo "########## $MODEL ##########"
    python3 -u scripts/run_full_dataset_dl.py --model $MODEL
    echo
done
