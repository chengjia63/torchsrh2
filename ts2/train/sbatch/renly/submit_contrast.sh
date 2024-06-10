#!/bin/bash

#SBATCH --job-name=exp
#SBATCH --mail-type=ALL
#SBATCH --output=slurm_out/%A_%a.out

# ---------------------------------------------------
# SBATCH --account=precisionhealth_owned1
# SBATCH --partition=precisionhealth
# SBATCH --cpus-per-task=5
# SBATCH --mem-per-cpu=19200m
# ----------------------------------------------------

# ----------------------------------------------------
# SBATCH --partition=spgpu
# SBATCH --account=bleu99
# SBATCH --cpus-per-task=4
# SBATCH --mem-per-cpu=4800m
# ----------------------------------------------------
#SBATCH --partition=tocho
#SBATCH --account=tocho_owned1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=7739m
# ---------------------------------------------------
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --gpu_cmode=shared
#SBATCH --time=2-00:00:00
# ---------------------------------------------------
# SBATCH --nodelist=armis26001
# ---------------------------------------------------
#SBATCH --array=0-1


source activate ts2

#srun python main.py -c config/renly/train_simclr_srh.yaml
#srun python main.py -c config/renly/train_dino_tcga.yaml
srun python main.py -c config/renly/train_ijepa_tcga.yaml
#python main.py -c config/renly/train_pjepa_tcga.yaml
