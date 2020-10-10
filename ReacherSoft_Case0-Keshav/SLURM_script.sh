#!/bin/bash
#SBATCH --job-name="vpg_hyperparameter_test"
#SBATCH --output="vpg_hyperparameter_test.out"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --export=ALL
#SBATCH -t 72:00:00

module load anaconda3
source activate venv

python3 reservoir_rl_spinning_up.py
