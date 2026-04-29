#!/usr/bin/env bash
#SBATCH --job-name=ts3_slide_embedding
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

mkdir -p logs/slurm

GRID_KEYS=(
  "++lightning_module.optimizer.lr"
  "++data.splits.train.dataloader.batch_size"
)

GRID_VALUES=(
  "1.0e-4 3.0e-4"
  "8 16"
)

COMBINATIONS=("")
for param_index in "${!GRID_KEYS[@]}"; do
  key="${GRID_KEYS[param_index]}"
  read -r -a values <<< "${GRID_VALUES[param_index]}"
  previous_combinations=("${COMBINATIONS[@]}")
  COMBINATIONS=()

  for prefix in "${previous_combinations[@]}"; do
    for value in "${values[@]}"; do
      if [[ -n "${prefix}" ]]; then
        COMBINATIONS+=("${prefix}"$'\n'"${key}=${value}")
      else
        COMBINATIONS+=("${key}=${value}")
      fi
    done
  done
done

num_jobs="${#COMBINATIONS[@]}"
task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set by sbatch --array}"

if (( task_id < 0 || task_id >= num_jobs )); then
  echo "Task id ${task_id} is out of range for ${num_jobs} grid jobs" >&2
  echo "Update #SBATCH --array to 0-$((num_jobs - 1))" >&2
  exit 1
fi

mapfile -t GRID_OVERRIDES <<< "${COMBINATIONS[task_id]}"

python -m ts3.train.main --config-name config/slide_embedding \
  "${GRID_OVERRIDES[@]}"
