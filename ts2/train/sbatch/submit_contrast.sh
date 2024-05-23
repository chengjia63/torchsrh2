#!/bin/bash

#SBATCH --job-name=exp
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --output=slurm_out/%A_%a.out

#SBATCH --account=tocho_owned1
#SBATCH --time=14-00:00:00

# SBATCH --account=precisionhealth_owned1
# SBATCH --partition=precisionhealth
# SBATCH --cpus-per-task=5
# SBATCH --mem-per-cpu=19200m
# SBATCH --ntasks-per-node=1
# SBATCH --gres=gpu:1

#SBATCH --partition=tocho
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=7739m
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
# SBATCH --gpu_cmode=shared

# SBATCH --nodelist=armis28003

#SBATCH --array=0-2

srun python train_contrastive.py -c=config/train_simclr_tcga.yaml
