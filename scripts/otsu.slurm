#!/bin/sh
#SBATCH --mem=80G
#SBATCH --job-name="otsu"
#SBATCH --time=00:20:00


python /home/lcur1043/pcam/scripts/otsu.py
