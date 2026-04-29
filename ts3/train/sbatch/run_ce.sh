#!/usr/bin/env bash
set -euo pipefail

sanitize_tag_value() {
  tr -cd '[:alnum:]' <<< "$1"
}

LR_VALUES=(1.0e-7 1.0e-6 1.0e-5 1.0e-4)
LR_VALUES=(1.0e-5)
BS_VALUES=(128 256)

COMBINATIONS=()

for lr in "${LR_VALUES[@]}"; do
  for bs in "${BS_VALUES[@]}"; do
    run_name="lr$(sanitize_tag_value "$lr")_bs$(sanitize_tag_value "$bs")"
    opts=(
      "++infra.tune_comment=${run_name}"
      "++meta_arch.optimizer.lr=${lr}"
      "++data.splits.train.dataloader.batch_size=${bs}"
    )
    COMBINATIONS+=("${opts[*]}")
  done
done

for i in {0..15}; do  # e.g. 0 2 3 or {0..3}
  echo "Running combination $i: ${COMBINATIONS[$i]}"

  HYDRA_FULL_ERROR=1 srun python -m ts3.train.main \
    --config-name config/slide_embedding_ce \
    ++infra.exp_name=ts3/silica_mil_ce \
    ++infra.comment=ce \
    ${COMBINATIONS[$i]}
done
