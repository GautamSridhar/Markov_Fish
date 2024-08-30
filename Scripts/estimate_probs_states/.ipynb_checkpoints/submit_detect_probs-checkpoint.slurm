#!/bin/bash
#SBATCH -p compute
#SBATCH -t 01:00:00
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=124G                                    
#SBATCH --output=out/out_%A_%a.out
#SBATCH --error=out/err_%A_%a.out


module load python/3.7.3

python3 -u estimate_detect_probs.py -idx ${SLURM_ARRAY_TASK_ID} -len_sim 1000 -n_bouts 1000
