#!/bin/bash

#------------------------------------------------------------------------------
#SBATCH --job-name=exp
#SBATCH --mail-type=ALL
#SBATCH --output=slurm_out/%A_%a.out
#------------------------------------------------------------------------------
# SBATCH --partition=precisionhealth
# SBATCH --cpus-per-task=5
# SBATCH --mem-per-cpu=18750m
# --------
# SBATCH --partition=gpu
# SBATCH --mem-per-cpu=7500m
# SBATCH --cpus-per-task=4
# --------
# SBATCH --partition=spgpu,gpu_mig40
# SBATCH --cpus-per-task=4
# SBATCH --mem-per-cpu=11630m
# --------
# SBATCH --partition=gpu_mig40
# SBATCH --cpus-per-task=8
# SBATCH --mem-per-cpu=16000m
# --------
#SBATCH --partition=tocho
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=7739m
#------------------------------------------------------------------------------
# SBATCH --account=tocho1
# SBATCH --account=tocho0
# SBATCH --account=precisionhealth_owned1
#SBATCH --account=tocho_owned1
#------------------------------------------------------------------------------
# SBATCH --nodelist=armis28003
#------------------------------------------------------------------------------
#SBATCH --time=14-00:00:00
# SBATCH --time=27:00:00
# SBATCH --time=16:00:00
#------------------------------------------------------------------------------
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#------------------------------------------------------------------------------
#SBATCH --array=0-3
#------------------------------------------------------------------------------

set -x
set -e

srun python main.py -c=config/chengjia/train_pjepa_tcga.yaml
#srun python main.py -c=config/chengjia/train_simclr_tcga.yaml
