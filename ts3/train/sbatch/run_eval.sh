#!/usr/bin/env bash
set -euo pipefail

ckpts=(
"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral/slocal__sharedhead_lr10e5_bs128_Apr29-05-06-01_2c069a82/models/epochepoch\=09.ckpt"
"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral/slocal__sharedhead_lr10e5_bs16_Apr29-04-06-46_5426a9b9/models/epochepoch\=09.ckpt"
"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral/slocal__sharedhead_lr10e5_bs32_Apr29-04-16-43_5c9e53dc/models/epochepoch\=09.ckpt"
"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral/slocal__sharedhead_lr10e5_bs64_Apr29-04-34-06_be0634c6/models/epochepoch\=09.ckpt"
"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral/slocal__sharedhead_lr50e4_bs128_Apr29-07-09-28_6c3e4344/models/epochepoch\=09.ckpt"
"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral/slocal__sharedhead_lr50e4_bs16_Apr29-06-10-12_67770460/models/epochepoch\=09.ckpt"
"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral/slocal__sharedhead_lr50e4_bs32_Apr29-06-20-08_24efb4dd/models/epochepoch\=09.ckpt"
"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral/slocal__sharedhead_lr50e4_bs64_Apr29-06-37-30_d7d506d9/models/epochepoch\=09.ckpt"
"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral/slocal__sharedhead_lr50e5_bs128_Apr29-03-02-48_ef154e0b/models/epochepoch\=09.ckpt"
"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral/slocal__sharedhead_lr50e5_bs16_Apr29-02-01-01_05716dac/models/epochepoch\=09.ckpt"
"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral/slocal__sharedhead_lr50e5_bs32_Apr29-02-11-33_14110e73/models/epochepoch\=09.ckpt"
"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral/slocal__sharedhead_lr50e5_bs64_Apr29-02-29-46_4c330d3e/models/epochepoch\=09.ckpt"
)



for ckpt in "${ckpts[@]}"; do
  echo "Running inference on: $ckpt"

  python -m ts3.train.infer --config-name config/slide_embedding_infer \
    ++inference.checkpoint_path="$ckpt"
done
