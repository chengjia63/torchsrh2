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
#SBATCH --mem-per-cpu=6g
#7739m
#------------------------------------------------------------------------------
# SBATCH --account=tocho1
# SBATCH --account=tocho0
# SBATCH --account=precisionhealth_owned1
#SBATCH --account=tocho_owned1
#------------------------------------------------------------------------------
# SBATCH --nodelist=armis26004
# SBATCH --exclude=armis2600[0-1]
#------------------------------------------------------------------------------
#SBATCH --time=5-00:00:00
# SBATCH --time=27:00:00
# SBATCH --time=16:00:00
#------------------------------------------------------------------------------
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#------------------------------------------------------------------------------
#SBATCH --array=0-1
#------------------------------------------------------------------------------

set -x
set -e

#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export TORCH_NCCL_TRACE_BUFFER_SIZE=16777216
#export CUDA_LAUNCH_BLOCKING=1

#srun python main.py -c=config/chengjia/train_pjepa_tcga.yaml
#srun python main.py -c=config/chengjia/train_modsimclr_tcga.yaml
#srun python main.py -c=config/chengjia/train_simclr_tcga.yaml
#srun python main.py -c=config/chengjia/train_committee.yaml
#srun python main.py -c=config/chengjia/train_committe_crd.yaml
#srun python main.py -c=config/chengjia/train_supcon_dbta28.yaml
#srun python main.py -c=config/chengjia/train_supcon_umbtb.yaml 
#srun python main_dinov2.py -c=config/chengjia/train_dinov2.yaml
#srun python main_dinov2.py -c=config/chengjia/train_dinov2_tryresume.yaml
#srun python main_dinov2.py -c=config/chengjia/finetune_dinov2.yaml
#srun python main_dinov2.py -c=config/chengjia/hidisc_finetune_dinov2.yaml

#srun python main_dinov2.py -c=config/chengjia/train_dinov2_scsrh.yaml
#srun python main_dinov2.py -c=config/chengjia/train_dinov2_scsrh_vitb.yaml

#srun python main.py -c=config/chengjia/train_supcon_scsrh7_debug.yaml
#srun python main.py -c=config/chengjia/train_mcm_srh7.yaml 
#srun python main.py -c=config/chengjia/train_ibot_scsrh7.yaml

#srun python main.py -c=config/chengjia/train_mcm_dinov2_srh7.yaml

#srun python main_dinov2.py -c=config/chengjia/train_mcm_dinov2_fmi.yaml
#srun python main_dinov2.py -c=config/chengjia/train_mcm_dinov2_fmi_nonorm.yaml

#srun python main_dinov2.py -c=config/chengjia/train_mcm_dinov2_fmitds_touint8.yaml
#srun python main_dinov2.py -c=config/chengjia/train_mcm_dinov2_fmitds_finetuned.yaml

#srun python main_dinov2.py -c=config/chengjia/train_mcmmcm_dinov2_fmitds_finetuned.yaml
#srun python main_dinov2.py -c=config/chengjia/train_pic.yaml
#srun python main_dinov2.py -c=config/chengjia/train_mcm_dinov2_fmitds_nouint8.yaml
#srun python main.py -c=config/chengjia/train_supcon_scsrh7.yaml
#srun python main.py -c=config/chengjia/train_supcon_scsrh7_orig.yaml

#srun python main_dinov2.py -c=config/chengjia/train_dinov2_cc.yaml

#srun python main_dinov2.py -c=config/chengjia/train_dinov2_tile_cellp.yaml
#srun python main_dinov2.py -c=config/chengjia/train_dinov2_tile_cellp_new.yaml
srun python main_dinov2.py -c=config/chengjia/train_dinov2_tile_cellp_new_aggressive.yaml
#srun python main_dinov2.py -c=config/chengjia/train_dinov2_cc_new.yaml
#srun python main_dinov2.py -c=config/chengjia/train_dinov2_ncc.yaml
#srun python main_dinov2.py -c=config/chengjia/train_dinov2_cc_noloco.yaml
#srun python main_dinov2.py -c=config/chengjia/tile/train_mcm_dinov2_fmitds_nouint8.yaml
