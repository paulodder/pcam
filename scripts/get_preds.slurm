#!/bin/bash
#SBATCH --mem=60G
#SBATCH --job-name="get_preds"
#SBATCH --time=48:00:00
#SBATCH --partition gpu_shared_course
#SBATCH --gpus-per-node=1
#SBATCH --output=/home/lcur1060/slurm/get_preds_%A.out

python /home/lcur1060/pcam/src/run_config.py
