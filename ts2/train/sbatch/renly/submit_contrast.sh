#!/bin/bash

#SBATCH --job-name=exp
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --output=slurm_out/%A_%a.out

#SBATCH --account=bleu99
#SBATCH --time=2-00:00:00

# SBATCH --account=precisionhealth_owned1
# SBATCH --partition=precisionhealth
# SBATCH --cpus-per-task=5
# SBATCH --mem-per-cpu=19200m
# SBATCH --ntasks-per-node=1
# SBATCH --gres=gpu:1

#SBATCH --partition=spgpu
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4800m
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --gpu_cmode=shared

# SBATCH --nodelist=armis26001

<<<<<<< Updated upstream
#SBATCH --array=0-2
source activate ts2

#srun python main.py -c config/renly/train_simclr_srh.yaml
#python main.py -c config/renly/train_ijepa_tcga.yaml
srun python main.py -c config/renly/train_ijepa_srh.yaml
=======
#SBATCH --array=3-3
source activate ts2

#srun python main.py -c config/renly/train_simclr_srh.yaml
#srun python main.py -c config/renly/train_dino_tcga.yaml
python main.py -c config/renly/train_ijepa_tcga.yaml
>>>>>>> Stashed changes
#python main.py -c config/renly/train_pjepa_tcga.yaml
