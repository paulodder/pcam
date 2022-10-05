#!/bin/bash
#SBATCH --mem=80G
#SBATCH --job-name="otsu"
#SBATCH --time=00:20:00


python $HOME/pcam/scripts/otsu.py
