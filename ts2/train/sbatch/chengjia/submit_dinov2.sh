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
# SBATCH --nodelist=armis26005
# SBATCH --exclude=armis2600[0-1]
#------------------------------------------------------------------------------
#SBATCH --time=5-00:00:00
# SBATCH --time=27:00:00
# SBATCH --time=16:00:00
#------------------------------------------------------------------------------
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#------------------------------------------------------------------------------
#SBATCH --array=0
#------------------------------------------------------------------------------

set -x
set -e

#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export TORCH_NCCL_TRACE_BUFFER_SIZE=16777216
#export CUDA_LAUNCH_BLOCKING=1

srun python main_dinov2.py -c=config/chengjia/train_dinov2_tile_cellp.yaml
