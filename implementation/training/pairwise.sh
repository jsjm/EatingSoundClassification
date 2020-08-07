#!/bin/sh
#SBATCH --job-name=matrix
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=4gb
#SBATCH --array 0-189

module load Pillow/6.2.1-GCCcore-8.3.0
module load binutils/2.31.1-GCCcore-8.2.0
module load Keras/2.2.4-foss-2019a-Python-3.7.2

python pairwise.py $SLURM_ARRAY_TASK_ID

