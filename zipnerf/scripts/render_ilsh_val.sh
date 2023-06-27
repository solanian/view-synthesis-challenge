#!/bin/bash

SCENE=004_00
EXPERIMENT=ilsh/devPhase_01/"$SCENE/factor_2_wo_recen_train_all"
DATA_ROOT=data/devPhase/devPhase_01
DATA_DIR="$DATA_ROOT"/"$SCENE"

# rm -rf exp/"$EXPERIMENT"/*
# CUDA_VISIBLE_DEVICES="4" accelerate launch \
CUDA_VISIBLE_DEVICES="0" python3 \
 render_val_pose.py --gin_configs=configs/ilsh.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
  --gin_bindings="Config.llff_use_all_images_for_testing = True" \
  --gin_bindings="Config.aug_yaw = 2" \
#   --gin_bindings="Config.aug_pitch = 2" \
#   --gin_bindings="Config.aug_pitch = 2" \