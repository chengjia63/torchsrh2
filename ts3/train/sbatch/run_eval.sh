#!/usr/bin/env bash
set -euo pipefail

ckpts=(
#"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_lin_celldropout_tune/slocal__rtamil_drop2575_lr10e4_bs16_May04-17-39-37_870689fc/models/epoch09.ckpt"
#"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_lin_celldropout_tune/slocal__rtamil_drop2575_lr10e4_bs32_May04-17-58-14_a684df25/models/epoch09.ckpt"
#"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_lin_celldropout_tune/slocal__rtamil_drop2575_lr10e4_bs64_May04-18-29-55_d030bab3/models/epoch09.ckpt"
#"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_lin_celldropout_tune/slocal__rtamil_drop2575_lr10e5_bs16_May04-17-28-54_8e2835d4/models/epoch09.ckpt"
#"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_lin_celldropout_tune/slocal__rtamil_drop2575_lr10e5_bs32_May04-17-39-37_aa463a37/models/epoch09.ckpt"
#"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_lin_celldropout_tune/slocal__rtamil_drop2575_lr10e5_bs64_May04-18-29-55_90fef0d5/models/epoch09.ckpt"
#"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_lin_celldropout_tune/slocal__rtamil_drop2575_lr10e6_bs16_May04-17-28-54_33c0a0d9/models/epoch09.ckpt"
#"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_lin_celldropout_tune/slocal__rtamil_drop2575_lr10e6_bs32_May04-17-39-37_742a1c44/models/epoch09.ckpt"
#"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_lin_celldropout_tune/slocal__rtamil_drop2575_lr10e6_bs64_May04-17-58-14_d3451973/models/epoch09.ckpt"
#"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_lin_celldropout_tune/slocal__rtamil_drop2575_lr50e5_bs16_May04-17-28-54_48615743/models/epoch09.ckpt"
#"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_lin_celldropout_tune/slocal__rtamil_drop2575_lr50e5_bs32_May04-17-58-14_56dbfedd/models/epoch09.ckpt"
#"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_lin_celldropout_tune/slocal__rtamil_drop2575_lr50e5_bs64_May04-18-29-55_bc613189/models/epoch09.ckpt"
#"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_lin_celldropout_tune/slocal__rtamil_drop2575_lr50e6_bs16_May04-17-28-54_4755e0d4/models/epoch09.ckpt"
#"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_lin_celldropout_tune/slocal__rtamil_drop2575_lr50e6_bs32_May04-17-39-37_9b62a575/models/epoch09.ckpt"
#"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_lin_celldropout_tune/slocal__rtamil_drop2575_lr50e6_bs64_May04-17-58-14_6a59f261/models/epoch09.ckpt"
#"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_lin_celldropout_tune/slocal__rtamil_pos_drop2575_lr10e4_bs16_May04-19-09-26_9f2db686/models/epoch09.ckpt"
#"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_lin_celldropout_tune/slocal__rtamil_pos_drop2575_lr10e4_bs32_May04-19-32-41_0d14f1d2/models/epoch09.ckpt"
#"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_lin_celldropout_tune/slocal__rtamil_pos_drop2575_lr10e4_bs64_May04-20-14-40_6a70fb73/models/epoch09.ckpt"
#"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_lin_celldropout_tune/slocal__rtamil_pos_drop2575_lr10e5_bs16_May04-18-56-11_ca8ba891/models/epoch09.ckpt"
#"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_lin_celldropout_tune/slocal__rtamil_pos_drop2575_lr10e5_bs32_May04-19-09-26_fbe12771/models/epoch09.ckpt"
#"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_lin_celldropout_tune/slocal__rtamil_pos_drop2575_lr10e5_bs64_May04-20-14-40_a612fca4/models/epoch09.ckpt"
#"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_lin_celldropout_tune/slocal__rtamil_pos_drop2575_lr10e6_bs16_May04-18-56-11_cdb0fc3e/models/epoch09.ckpt"
#"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_lin_celldropout_tune/slocal__rtamil_pos_drop2575_lr10e6_bs32_May04-19-09-26_1ea7a53a/models/epoch09.ckpt"
#"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_lin_celldropout_tune/slocal__rtamil_pos_drop2575_lr10e6_bs64_May04-19-32-41_d10b9e67/models/epoch09.ckpt"
#"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_lin_celldropout_tune/slocal__rtamil_pos_drop2575_lr50e5_bs16_May04-18-56-11_2dc8d5cc/models/epoch09.ckpt"
#"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_lin_celldropout_tune/slocal__rtamil_pos_drop2575_lr50e5_bs32_May04-19-32-41_46fd979b/models/epoch09.ckpt"
#"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_lin_celldropout_tune/slocal__rtamil_pos_drop2575_lr50e5_bs64_May04-20-14-40_57c43198/models/epoch09.ckpt"
#"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_lin_celldropout_tune/slocal__rtamil_pos_drop2575_lr50e6_bs16_May04-18-56-11_6900376d/models/epoch09.ckpt"
#"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_lin_celldropout_tune/slocal__rtamil_pos_drop2575_lr50e6_bs32_May04-19-09-26_9aa75746/models/epoch09.ckpt"
#"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_coral_lin_celldropout_tune/slocal__rtamil_pos_drop2575_lr50e6_bs64_May04-19-32-41_d50bd45c/models/epoch09.ckpt"

"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_panther_coral_1024/slocal__panther_lr10e3_bs32_May05-17-07-31_28a2eb55/models/epoch09.ckpt"
"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_panther_coral_1024/slocal__panther_lr10e4_bs32_May05-17-07-31_6cacf664/models/epoch09.ckpt"
"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_panther_coral_1024/slocal__panther_lr10e5_bs32_May05-16-15-31_240d3047/models/epoch09.ckpt"
"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_panther_coral_1024/slocal__panther_lr10e6_bs32_May05-16-15-31_0e827ed6/models/epoch09.ckpt"
"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_panther_coral_1024/slocal__panther_lr10e7_bs32_May05-16-15-31_611c00a5/models/epoch09.ckpt"
"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_panther_coral_1024/slocal__panther_lr50e4_bs32_May05-17-07-31_f740a2b1/models/epoch09.ckpt"
"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_panther_coral_1024/slocal__panther_lr50e5_bs32_May05-17-07-31_c0200595/models/epoch09.ckpt"
"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts3/fix_patch/silica_mil_panther_coral_1024/slocal__panther_lr50e6_bs32_May05-16-15-31_ad65af1b/models/epoch09.ckpt"

)


gpu=0
for ckpt in "${ckpts[@]}"; do
  echo "GPU $gpu | Running inference on: $ckpt"

  #CUDA_VISIBLE_DEVICES=$gpu python -m ts3.train.eval --config-name config/eval_rtamil \
  #  ++inference.checkpoint_path="$ckpt" &

  #CUDA_VISIBLE_DEVICES=$gpu python -m ts3.train.eval --config-name config/eval_rtamil_pos \
  #  ++inference.checkpoint_path="$ckpt" &


  CUDA_VISIBLE_DEVICES=$gpu python -m ts3.train.eval --config-name config/eval_panther_coral \
    ++inference.checkpoint_path="$ckpt" &

  gpu=$(( (gpu + 1) % 4 ))
  if [[ $gpu -eq 0 ]]; then
    wait
  fi
done
wait
