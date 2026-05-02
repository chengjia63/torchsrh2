#!/usr/bin/env bash
set -euo pipefail

ckpts=(
'/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_mlp_tune/slocal__rtamil_lr10e4_bs16_May01-03-39-18_757950a3/models/epochepoch\=09.ckpt'
'/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_mlp_tune/slocal__rtamil_lr10e4_bs32_May01-03-55-44_6d6b7f29/models/epochepoch\=09.ckpt'
'/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_mlp_tune/slocal__rtamil_lr10e4_bs64_May01-04-17-18_f03f475d/models/epochepoch\=09.ckpt'
'/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_mlp_tune/slocal__rtamil_lr10e5_bs16_May01-00-42-25_70e9fa61/models/epochepoch\=09.ckpt'
'/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_mlp_tune/slocal__rtamil_lr10e5_bs32_May01-01-07-51_e77bec3c/models/epochepoch\=09.ckpt'
'/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_mlp_tune/slocal__rtamil_lr10e5_bs64_May01-01-41-10_2abfe0cd/models/epochepoch\=09.ckpt'
'/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_mlp_tune/slocal__rtamil_lr10e6_bs16_May01-03-53-52_5950d245/models/epochepoch\=09.ckpt'
'/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_mlp_tune/slocal__rtamil_lr10e6_bs32_May01-04-08-46_51eb3328/models/epochepoch\=09.ckpt'
'/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_mlp_tune/slocal__rtamil_lr10e6_bs64_May01-04-30-00_317bbc5c/models/epochepoch\=09.ckpt'
'/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_mlp_tune/slocal__rtamil_lr50e4_bs16_May01-02-21-21_82aff3d3/models/epochepoch\=09.ckpt'
'/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_mlp_tune/slocal__rtamil_lr50e4_bs32_May01-02-35-19_ccd475d8/models/epochepoch\=09.ckpt'
'/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_mlp_tune/slocal__rtamil_lr50e4_bs64_May01-02-57-28_c5e8bfff/models/epochepoch\=09.ckpt'
'/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_mlp_tune/slocal__rtamil_lr50e5_bs32_Apr30-23-30-44_b392d3d2/models/epochepoch\=09.ckpt'
'/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_mlp_tune/slocal__rtamil_lr50e5_bs64_Apr30-23-50-12_5782b542/models/epochepoch\=09.ckpt'
'/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_mlp_tune/slocal__rtamil_lr50e6_bs16_May01-15-35-17_efd23519/models/epochepoch\=09.ckpt'
'/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_mlp_tune/slocal__rtamil_lr50e6_bs32_May01-02-51-19_2c92af49/models/epochepoch\=09.ckpt'
'/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_mlp_tune/slocal__rtamil_lr50e6_bs64_May01-03-12-35_4c03dea9/models/epochepoch\=09.ckpt'
)


gpu=0
for ckpt in "${ckpts[@]}"; do
  echo "GPU $gpu | Running inference on: $ckpt"

  CUDA_VISIBLE_DEVICES=$gpu python -m ts3.train.eval --config-name config/slide_embedding_infer \
    ++inference.checkpoint_path="$ckpt" &

  gpu=$(( (gpu + 1) % 4 ))
  if [[ $gpu -eq 0 ]]; then
    wait
  fi
done
wait
