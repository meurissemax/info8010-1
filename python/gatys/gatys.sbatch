#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=mmeurisse-gatys     # Name of the job 
#SBATCH --export=ALL                   # Export all environment variables
#SBATCH --output=mmeurisse-gatys.log   # Log-file (important!)
#SBATCH --cpus-per-task=2              # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4G               # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                   # Number of GPU's
#SBATCH --time=1:00:00                 # Max execution time
#

# Activate the anaconda environnement
source miniconda3/etc/profile.d/conda.sh
conda activate neural_style_transfer

# Run the script
cd /home/mmeurisse/info8010-1/python/gatys/
python main.py
