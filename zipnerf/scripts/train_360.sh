#!/bin/bash

SCENE=bicycle
EXPERIMENT=360_v2/"$SCENE/sampled_step_9"
DATA_ROOT=data/mip_nerf_360
DATA_DIR="$DATA_ROOT"/"$SCENE"

# rm exp/"$EXPERIMENT"/*
CUDA_VISIBLE_DEVICES="4" \
# accelerate launch \
python3 train.py --gin_configs=configs/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
  --gin_bindings="Config.factor = 4"
