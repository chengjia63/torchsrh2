#!/usr/bin/env bash
set -euo pipefail

SITE_RES_DIR='/nfs/turbo/umms-tocho/code/chengjia/torchsrh2/ts2/utils/silica_eval/infil/site_res'

eval_dirs=(
#"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_lin_tune/slocal__rtamil_lr10e5_bs64_May02-00-53-02_6201e950/evals/trainval_May03-02-46-54_53f3999c"

"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_panther_coral_1024/slocal__panther_lr10e4_bs32_May05-17-07-31_6cacf664/evals/trainval_May06-16-50-24_09efbf6d"
)

for eval_dir in "${eval_dirs[@]}"; do
  echo "Generating portal for: $eval_dir"
  python -m ts3.train.portal "$eval_dir" "$SITE_RES_DIR"
done
