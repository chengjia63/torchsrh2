#!/usr/bin/env bash
set -euo pipefail

sanitize_tag_value() {
  tr -cd '[:alnum:]' <<< "$1"
}

LR_VALUES=(1.0e-7 1.0e-6 5.0e-6 1.0e-5 5.0e-5 1.0e-4 5.0e-4 1.0e-3)
BS_VALUES=(32)

# (16 32 64)


COMBINATIONS=()

#"++infra.exp_name=ts3/fix_patch/silica_mil_coral_lin_celldropout_tune"

for bs in "${BS_VALUES[@]}"; do
  for lr in "${LR_VALUES[@]}"; do
    run_name="lr$(sanitize_tag_value "$lr")_bs$(sanitize_tag_value "$bs")"
    opts=(
      "++infra.exp_name=ts3/fix_patch/silica_mil_panther_coral_1024"
      "++infra.tune_comment=${run_name}"
      "++meta_arch.optimizer.lr=${lr}"
      "++data.splits.train.dataloader.batch_size=${bs}"
    )
    COMBINATIONS+=("${opts[*]}")
  done
done

gpu=0
for i in "${!COMBINATIONS[@]}"; do
  echo "GPU $gpu | Running combination $i: ${COMBINATIONS[$i]}"

  # HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=$gpu srun python -m ts3.train.main --config-name config/abmil_coral ${COMBINATIONS[$i]} &

  # HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=$gpu python -m ts3.train.main --config-name config/train_rtamil_coral ${COMBINATIONS[$i]} &

  HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=$gpu python -m ts3.train.main --config-name  config/train_panther_coral  ${COMBINATIONS[$i]} &
  gpu=$(( (gpu + 1) % 4 ))
  if [[ $gpu -eq 0 ]]; then
    wait
  fi
done
wait

