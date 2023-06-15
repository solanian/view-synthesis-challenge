#!/bin/bash

## Starting Kit
parent_dir_downloaded_sub_dataset="/home/ILSH_DB" # Set the directory containing the zip files
starting_kit_path="/home/ILSH_DB/starting_kit_ILSH"
phasename="devPhase" ## choose between devPhase and chaPhase

# Find and extract all zip files in the directory
find "$parent_dir_downloaded_sub_dataset" -name "$phasename*.zip" -exec unzip -d "$parent_dir_downloaded_sub_dataset" {} \;
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' '-'

python 1_lightstage_subdataset_restructure.py --source_dir $parent_dir_downloaded_sub_dataset --target_dir $parent_dir_downloaded_sub_dataset/$phasename --phase_in $phasename
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' '-'

python 2_loading_toy_example_and_applying_mask.py --eval_ex_imgs_dir_path $starting_kit_path/eval_ex_imgs 
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' '-'

python 3_print_required_file_names_in_the_current_phase.py --split_info_file $parent_dir_downloaded_sub_dataset/$phasename/challenge_data_split_$phasename.json
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' '-'

python 4_create_dummy_submission_zip.py --split_info_file $parent_dir_downloaded_sub_dataset/$phasename/challenge_data_split_$phasename.json --target_dir $parent_dir_downloaded_sub_dataset/dummyDir
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' '-'

python 5_evaluation_script_example.py --eval_ex_imgs_dir_path $starting_kit_path/eval_ex_imgs 
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' '-'

python 6_examine_submission_file.py --sub_file $parent_dir_downloaded_sub_dataset/dummy_submission.zip --split_info_file $parent_dir_downloaded_sub_dataset/$phasename/challenge_data_split_$phasename.json
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' '-'

python tip1_visualise_camera_poses.py --in_dir $parent_dir_downloaded_sub_dataset/$phasename/004_00 --out_path $parent_dir_downloaded_sub_dataset/camera_poses.mp4 --save_jpg False
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' '-'