#!/bin/bash

set -x
set -e


for i in {0..2}; do
  echo $i
  SLURM_ARRAY_TASK_ID=$i python train_contrastive.py -c=config/train_simclr_tcga.yaml 
done
