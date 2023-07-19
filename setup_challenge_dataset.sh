#!/bin/bash
DATA_ROOT=data_root/chaPhase

gdown 1iRXdxKNfVpGCUbKtVNbrr95i8L_RdICf -O $DATA_ROOT/chaPhase_01.zip # chaPhase_01.zip
gdown 1IU9gyO-PFo8eODh_TNGJIB3WpMWq2KCt -O $DATA_ROOT/chaPhase_02.zip # chaPhase_02.zip

unzip $DATA_ROOT/chaPhase_01.zip -d $DATA_ROOT
unzip $DATA_ROOT/chaPhase_02.zip -d $DATA_ROOT

rm -rf $DATA_ROOT/chaPhase_01.zip $DATA_ROOT/chaPhase_02.zip