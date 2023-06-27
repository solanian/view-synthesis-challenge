#!/bin/bash

SCENE=004_00
EXPERIMENT=ilsh/devPhase_01/"$SCENE/factor_2_bg_remove_wo_pt"
DATA_ROOT=data/devPhase/devPhase_01
DATA_DIR="$DATA_ROOT"/"$SCENE"

# rm -rf exp/"$EXPERIMENT"/*
# CUDA_VISIBLE_DEVICES="4" accelerate launch \
CUDA_VISIBLE_DEVICES="1" python3 \
 train.py --gin_configs=configs/ilsh.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \