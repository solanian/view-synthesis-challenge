#!/bin/bash

SCENE=004_00
EXPERIMENT=ilsh/devPhase_01/"$SCENE"_nf_8e-1_28-e1
DATA_ROOT=data/devPhase/devPhase_01
DATA_DIR="$DATA_ROOT"/"$SCENE"

rm exp/"$EXPERIMENT"/*
# accelerate launch \
CUDA_VISIBLE_DEVICES="5" python3 \
 train.py --gin_configs=configs/ilsh.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \