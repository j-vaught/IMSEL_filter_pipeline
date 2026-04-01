#!/bin/bash
#SBATCH --job-name=classical-full
#SBATCH --partition=gpu-H200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-classical-full-%j.out
#SBATCH --error=slurm-classical-full-%j.err

# Full dataset-wide ODS for all 54 classical edge detectors across 4 datasets,
# plus WVF/LF reference configs for direct comparison.

module load python3/anaconda/3.12

echo "Classical full-dataset ODS at $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

cd $HOME/edge-detection-filter-critique
pip install --user -e . 2>&1 | tail -1

python3 -u scripts/run_classical_full_dataset.py
