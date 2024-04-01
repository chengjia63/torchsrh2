#!/bin/bash

#SBATCH --job-name=exp
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --output=slurm_out/%A_%a.out

# SBATCH --account=precisionhealth_owned1
# SBATCH --partition=precisionhealth
# SBATCH --cpus-per-task=5
# SBATCH --mem-per-cpu=19200m
# SBATCH --ntasks-per-node=8
# SBATCH --gres=gpu:8

# SBATCH --account=tocho0
# SBATCH --partition=gpu
# SBATCH --cpus-per-task=20
# SBATCH --ntasks-per-node=1
# SBATCH --mem-per-cpu=4500m
# SBATCH --gres=gpu:1
# SBATCH --gpu_cmode=shared


#SBATCH --partition=spgpu
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=11630m
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
# SBATCH --gpu_cmode=shared

# SBATCH --account=eecs598s007f23_class
# SBATCH --time=16:00:00

#SBATCH --account=tocho1
#SBATCH --time=14-00:00:00

# SBATCH --nodelist=gl1517

sleep inf
