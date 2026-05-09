#!/usr/bin/env bash
set -exuo pipefail


start_task_id="4"
config_path="config/cell_inf_ucsf.yaml"

for gpu_id in 0 1 2 3; do
  task_id=$((start_task_id + gpu_id))
  log_path="ucsf_task_${task_id}_gpu_${gpu_id}.log"
  CUDA_VISIBLE_DEVICES="${gpu_id}" SLURM_ARRAY_TASK_ID="${task_id}" \
    python run_single_cell_inference.py -c="${config_path}" \
    > "${log_path}" 2>&1 &
done

wait
