#!/usr/bin/env bash
set -euo pipefail

SITE_RES_DIR='/nfs/turbo/umms-tocho/code/chengjia/torchsrh2/ts2/utils/silica_eval/infil/site_res'

eval_dirs=(
# '/path/to/train_run/evals/eval_name'
)

for eval_dir in "${eval_dirs[@]}"; do
  echo "Generating portal for: $eval_dir"
  python -m ts3.train.portal "$eval_dir" "$SITE_RES_DIR"
done
