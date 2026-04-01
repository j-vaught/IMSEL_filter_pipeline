#!/bin/bash
#SBATCH --job-name=aniso-lf-eval
#SBATCH --partition=gpu-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-aniso-lf-eval-%j.out
#SBATCH --error=slurm-aniso-lf-eval-%j.err

# Back-to-back evaluation: Original LF vs Anisotropic LF (conv2d + Triton)
# Runs on same GPU, same images, same process for minimal bias

module load python3/anaconda/3.12

echo "Anisotropic LF Evaluation at $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

cd $HOME/edge-detection-filter-critique
pip install --user -e . 2>&1 | tail -1
pip install --user triton 2>&1 | tail -1

python3 -u scripts/eval_aniso_lf.py --n-images 3 --n-runs 5

echo "Done at $(date)"
