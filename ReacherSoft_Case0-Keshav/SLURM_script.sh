#!/bin/bash
#SBATCH --job-name="spiking_512_seed_test"
#SBATCH --output="spiking_512_seed_test.out"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --export=ALL
#SBATCH -t 72:00:00

module load anaconda3
source activate //pylon5/mcz3atp/kshivvy/venv

python3 reservoir_rl_spinning_up.py
