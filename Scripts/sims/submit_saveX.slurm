#!/bin/bash
#SBATCH -p compute
#SBATCH -t 01:00:00
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G                                    
#SBATCH --output=out/out_s_%A.out
#SBATCH --error=out/err_s_%A.out


module load python/3.7.3

python3 -u save_X.py
