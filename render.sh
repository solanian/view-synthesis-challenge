#!/bin/bash

folder_path="/home/ubuntu/view-synthesis-challenge/TensoRF/configs/submit_result"
files_and_dirs=("$folder_path"/*)
num_elements=${#files_and_dirs[@]}
weight_path="/home/ubuntu/view-synthesis-challenge/TensoRF/log"

for file in "${files_and_dirs[@]}"
do
    # Extract the filename from the file path
    filename=$(basename "$file")
    # Construct the corresponding weight file path using weight_path and filename
    weight_file="$weight_path/$filename/$filename.th"

    python train.py --config "$file" --ckpt "$weight_file" --render_visual 1
done
