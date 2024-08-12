#!/bin/bash

#------------------------------------------------------------------------------
#SBATCH --job-name=infra
#SBATCH --mail-type=ALL
#SBATCH --output=slurm_out/infra/%A_%a.out
#------------------------------------------------------------------------------
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4g
# --------
# SBATCH --partition=precisionhealth
# SBATCH --cpus-per-task=5
# SBATCH --mem-per-cpu=18750m
# --------
# SBATCH --partition=gpu
# SBATCH --mem-per-cpu=4500m
# SBATCH --cpus-per-task=13
# 
# SBATCH --partition=gpu
# SBATCH --mem-per-cpu=7500m
# SBATCH --cpus-per-task=8
# --------
# SBATCH --partition=spgpu,gpu_mig40
# SBATCH --cpus-per-task=4
# SBATCH --mem-per-cpu=11630m
# --------
# SBATCH --partition=gpu_mig40
# SBATCH --cpus-per-task=8
# SBATCH --mem-per-cpu=16000m
# --------
# SBATCH --partition=tocho
# SBATCH --cpus-per-task=8
# SBATCH --mem-per-cpu=7739m
#------------------------------------------------------------------------------
#SBATCH --account=tocho1
# SBATCH --account=tocho0
# SBATCH --account=precisionhealth_owned1
# SBATCH --account=tocho_owned1
#------------------------------------------------------------------------------
# SBATCH --nodelist=armis20108
#------------------------------------------------------------------------------
# SBATCH --time=14-00:00:00
#SBATCH --time=01:00:00
#------------------------------------------------------------------------------
#SBATCH --nodes=1
# SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#------------------------------------------------------------------------------
#SBATCH --array=0-2619
#------------------------------------------------------------------------------

set -x
set -e

python convert_ts_embs.py