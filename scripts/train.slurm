#!/bin/sh
#SBATCH --mem=80G
#SBATCH --job-name="train_gcnn"
#SBATCH --time=24:00:00
#SBATCH --partition gpu_shared_course
#SBATCH --gpus-per-node=1


python /home/lcur1043/pcam/scripts/train.py
