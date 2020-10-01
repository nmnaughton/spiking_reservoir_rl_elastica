#!/bin/bash
#SBATCH --job-name="nengoRL"
#SBATCH --output="nengoRL.out"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --export=ALL
#SBATCH -t 24:00:00

module load anaconda3
source activate venv

python3 reservoir_rl.py
