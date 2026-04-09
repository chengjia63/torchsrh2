#!/bin/bash

set -x
set -e

for i in {0..0}; do
  SLURM_ARRAY_TASK_ID="$i" python main_cell_inference.py -c=config/chengjia/inference_dinov2_scsrhdb.yaml
  #SLURM_ARRAY_TASK_ID="$i" python main_cell_inference.py -c=config/chengjia/inference_dinov2aw_scsrhdb.yaml
done

for i in {0..10}; do
    SLURM_ARRAY_TASK_ID="$i" python main_cell_inference.py -c=config/chengjia/inference_dinov2_scsrh_test_perturbed.yaml
#  SLURM_ARRAY_TASK_ID="$i" python main_cell_inference.py -c=config/chengjia/inference_dinov2aw_scsrh_test_perturbed.yaml
done

for i in {0..21}; do
   SLURM_ARRAY_TASK_ID="$i" python main_cell_inference.py -c=config/chengjia/inference_dinov2_scsrh_paired.yaml
#  SLURM_ARRAY_TASK_ID="$i" python main_cell_inference.py -c=config/chengjia/inference_dinov2aw_scsrh_paired.yaml
done
