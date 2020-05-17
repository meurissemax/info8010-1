#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=gatys               # Name of the job 
#SBATCH --export=ALL                   # Export all environment variables
#SBATCH --output=gatys.log             # Log-file (important!)
#SBATCH --gres=gpu:1                   # Number of GPU's (per node)
#SBATCH --cpus-per-task=2              # CPU cores/threads
#SBATCH --mem=16G                      # Memory (per node)
#SBATCH --time=2:00:00                 # Max execution time
#

# Activate the anaconda environnement
source /home/mmeurisse/miniconda3/etc/profile.d/conda.sh
conda activate neural_style_transfer

# Run the script
cd /home/mmeurisse/info8010-1/gatys/
python main.py
