#!/bin/bash
DATA_ROOT=/workspace/view-synthesis-challenge/GeoNeRF/input_data/devPhase

gdown https://drive.google.com/uc?id=1qhe2GTjBuTmRkco2SFrwjVMa0Nho2nYx -O $DATA_ROOT/devPhase_01.zip # devPhase_01.zip
gdown https://drive.google.com/uc?id=1l1LEry5y7fn3Zpj_BjxvxWU6IV1H-Ey7 -O $DATA_ROOT/devPhase_02.zip # devPhase_02.zip

unzip $DATA_ROOT/devPhase_01.zip -d $DATA_ROOT
unzip $DATA_ROOT/devPhase_02.zip -d $DATA_ROOT

rm -rf $DATA_ROOT/devPhase_01.zip $DATA_ROOT/devPhase_02.zip