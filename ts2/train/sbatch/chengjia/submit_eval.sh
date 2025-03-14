#!/bin/bash

#------------------------------------------------------------------------------
#SBATCH --job-name=eval
#SBATCH --mail-type=ALL
#SBATCH --output=slurm_out/%A_%a.out
#------------------------------------------------------------------------------
# SBATCH --partition=precisionhealth
# SBATCH --cpus-per-task=5
# SBATCH --mem-per-cpu=18750m
# --------
# SBATCH --partition=gpu
# SBATCH --mem-per-cpu=4500m
# SBATCH --cpus-per-task=13
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
# SBATCH --nodelist=armis20108
#------------------------------------------------------------------------------
# SBATCH --time=1-00:00:00
#SBATCH --time=01:00:00
#------------------------------------------------------------------------------
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#------------------------------------------------------------------------------
#SBATCH --array=0-7
#------------------------------------------------------------------------------

echo @@@ EVALUATION @@@
set -x
set -e

#python main.py -c=config/chengjia/eval_scsrh7_dinov2.yaml
python main.py -c=config/chengjia/eval_scsrh7.yaml

#python main.py -c=config/chengjia/eval_dbta_supcon.yaml

#python main.py -c=config/chengjia/eval.yaml
#python main.py -c=config/chengjia/eval_glioma.yaml
#python main.py -c=config/chengjia/eval_umbtb.yaml
