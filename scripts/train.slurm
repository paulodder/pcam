#!/bin/bash
#SBATCH --mem=60G
#SBATCH --job-name="train_gcnn"
#SBATCH --time=24:00:00
#SBATCH --partition gpu_shared_course
#SBATCH --gpus-per-node=1


python $HOME/pcam/scripts/train.py
