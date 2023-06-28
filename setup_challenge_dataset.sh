#!/bin/bash
DATA_ROOT=/workspace/view-synthesis-challenge/data_root/devPhase

gdown 1wUztXKTFmP4g-yOuv3MuRBijz1S25r48 -O $DATA_ROOT/devPhase_01.zip # devPhase_01.zip
gdown 17aUOFbvci5Cd9_OUe6rpWp0vYRxPRvjJ -O $DATA_ROOT/devPhase_02.zip # devPhase_02.zip

unzip $DATA_ROOT/devPhase_01.zip -d $DATA_ROOT
unzip $DATA_ROOT/devPhase_02.zip -d $DATA_ROOT

rm -rf $DATA_ROOT/devPhase_01.zip $DATA_ROOT/devPhase_02.zip