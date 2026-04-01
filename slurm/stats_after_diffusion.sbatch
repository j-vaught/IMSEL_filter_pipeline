#!/bin/bash
#SBATCH --job-name=stats-v2
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-stats-v2-%j.out
#SBATCH --error=slurm-stats-v2-%j.err

# Re-run statistical analysis with corrected DiffusionEdge results.

module load python3/anaconda/3.12

echo "Statistical analysis (with DiffusionEdge) at $(date)"
echo "Node: $(hostname)"
echo

cd $HOME/edge-detection-filter-critique
pip install --user -e . 2>&1 | tail -1
pip install --user scipy 2>&1 | tail -1

# Delete old results so they get regenerated
rm -f outputs/statistical_analysis/results.json

python3 -u scripts/run_statistical_analysis.py
