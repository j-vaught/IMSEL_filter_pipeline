#!/bin/bash
#SBATCH --job-name=stats
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-stats-%j.out
#SBATCH --error=slurm-stats-%j.err

# Statistical analysis of cross-dataset edge detection performance.
# No GPU computation needed but uses H200 node for fast CPU + memory.

module load python3/anaconda/3.12

echo "Statistical analysis at $(date)"
echo "Node: $(hostname)"
echo

cd $HOME/edge-detection-filter-critique
pip install --user -e . 2>&1 | tail -1
pip install --user scipy 2>&1 | tail -1

python3 -u scripts/run_statistical_analysis.py
