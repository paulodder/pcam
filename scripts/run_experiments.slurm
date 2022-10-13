#!/bin/bash
#SBATCH --mem=60G
#SBATCH --job-name="run_exp"
#SBATCH --time=48:00:00
#SBATCH --partition gpu_shared_course
#SBATCH --gpus-per-node=1


python /home/lcur1043/pcam/scripts/run_experiments.py --experiment_name LAYERS
