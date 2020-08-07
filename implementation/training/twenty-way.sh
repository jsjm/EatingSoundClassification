#!/bin/sh
#SBATCH --job-name=uniform
#SBATCH --partition=cpu-medium
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=4gb
#SBATCH --array 0-9

module load Pillow/6.2.1-GCCcore-8.3.0
module load binutils/2.31.1-GCCcore-8.2.0
module load Keras/2.2.4-foss-2019a-Python-3.7.2

python uniform.py
python grouped.py
