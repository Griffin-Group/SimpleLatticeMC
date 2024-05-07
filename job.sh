#!/bin/bash
#SBATCH --job-name=mc1
#SBATCH --partition=etna
#SBATCH --account=nano
#SBATCH --qos=normal
#SBATCH --exclusive=user
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00

module load python/3.10.10
python main.py 

