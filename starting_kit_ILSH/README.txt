## Starting Kit

We provide this starting kit in addition to the dataset to help participants understand what to submit in each phase and to avoid mistakes before submission by examining the submission zip file on their local machine. We also provide an example scoring script to show how PSNR and SSIM are calculated.

There are mainly six script files to be used as follows:

1_lightstage_subdataset_restructure.py: This file is to combine the extracted sub-datasets and produce the devPhase/chaPhase dataset.

2_loading_toy_example_and_applying_mask.py: This file provides an example of calculating masked loss.

3_print_required_file_names_in_the_current_phase.py: This script can be used to check the list of files that need to be submitted depending on the phase. 

4_create_dummy_submission_zip.py: This script creates a zip file containing dummy (black) PNG files and a readme file in a folder that demonstrates a submission file.

5_evaluation_script_example.py: This file is a simple example of PSNR and SSIM calculation when there is a given submission file (a rendered image) and a pair of reference ground truth image and face mask. 

6_examine_submission_file.py: This file takes two arguments: the zip file path and the challenge_data_split_{devPhase, chaPhase}.json path. It examines whether the zip file contains the expected number of images and a readme file before submission. If you confirm that the zip file contains the proper files, you are now ready to submit your zip file to the CodaLab evaluation server.

tip1_visualise_camera_poses.py: This file demonstrates how to convert camera poses from Blender format to OpenCV format using the transforms_train.json input file. The output file visualizes the given camera poses in either JPEG or MP4 format.

* Please refer to the run_everything_at_once.sh file for the example arguments of each script.

--------------------------------
### Usage of 1_lightstage_subdataset_restructure.py

* Example command: 
> python 1_lightstage_subdataset_restructure.py --source_dir /path/to/<parentDirectory>_of_{devPhase_*}/ --target_dir ./<phaseName> --phase_in <phasename>

* Arguments explained: 
- source_dir: The path to the parent directory <PD> of all the extracted folders <PD>/devPhase_01/, <PD>/devPhase_02/, etc.
- target_dir: The path to the combining destination folder, such as <PD>/devPhase or <PD>/chaPhase.
- phase_in: The phase name should be selected between devPhase and chaphase. if the source folder contains  folders with names starting with devPhase_, then you should use devphase. Otherwise, use chaphase.

* Description:
This script helps create a combined dataset for either the Development Phase (devPhase) or the Challenge Phase (chaPhase).

--------------------------------
### Usage of 2_loading_toy_example_and_applying_mask.py

* Example command: 
> python 2_loading_toy_example_and_applying_mask.py --eval_ex_imgs_dir_path ./eval_ex_imgs 

* Arguments explained: 
- eval_ex_imgs_dir_path: The path to the eval_ex_imgs directory of the starting kit.

* Description:
This script loads images and border masks (not face masks, which will not be released to participants) to prevent training from being distracted by the black and empty border region. The script uses masked tensors to calculate loss (e.g., PSNR). Our initial investigation found that avoiding this region during training improves model quality.

--------------------------------
### Usage of 3_print_required_file_names_in_the_current_phase.py

* Example command: 
> python 3_print_required_file_names_in_the_current_phase.py --split_info_file ./<phaseName>/challenge_data_split_<phaseName>.json

* Arguments explained: 
- split_info_file: The path to the JSON file.

* Description: 
<phaseName> is either devPhase or chaPhase. if <phaseName> is devPhase, it returns all the names of validation files with readme.txt.
If it is chaPhase, it returns all the names of test files with readme.txt. The files listed by the script should be zipped into a single submission file.

--------------------------------
### Usage of 4_create_dummy_submission_zip.py

* Example command: 
> python 4_create_dummy_submission_zip.py --split_info_file ./<phaseName>/challenge_data_split_<phaseName>.json --target_dir ./dummyDir

* Arguments explained: 
- split_info_file: The path to the JSON file.
- target_dir: The path to the desired dummy directory where you would like to generate black image files and a readme text file within the folder.

* Description: 
This script outputs black images with names reflecting the corresponding submission files based on the phase-dependent JSON file. It also generates a single dummy_submission.zip file as an example of a submission file. You can replace only the first three images to test a subset of your results or replace all images to obtain official validation results through CodaLab. Make sure to include the correct information in the readme file.

--------------------------------
### Usage of 5_evaluation_script_example.py

* Example command: 
> python 5_evaluation_script_example.py --eval_ex_imgs_dir_path ./eval_ex_imgs 

* Arguments explained: 
- eval_ex_imgs_dir_path: The path to the eval_ex_imgs directory of the starting kit.

* Description:
This script evaluates submission files and returns PSNR and SSIM scores for all files, as well as for the first three files to provide faster feedback. To get scores for only the first three images, participants should submit black images for the remaining files in the zip file. This reduces the final zip file size and ultimately speeds up submission and calculation time since the script skips calculating black images.

--------------------------------
### Usage of 6_examine_submission_file.py

* Example command: 
> python 6_examine_submission_file.py --sub_file ./dummy_submission.zip --split_info_file ./<phaseName>/challenge_data_split_<phaseName>.json

* Arguments explained: 
- sub_file: The path to the submission zip file.
- split_info_file: The path to the JSON file.

* Description: 
<phaseName> is either devPhase or chaPhase, and at the same time, the zip file (sub_file) contents are examined depending on the phaseName. Thus, depending on the phase, it is important to put the correct <phaseName> for the JSON file path. Then, the script examines the name, resolution, and channel of images. In addition, it also checks the readme file contents. When there is a mistake, you can find it simply by using this script before submission. But even before using this script, make sure you are putting all the images and a readme file that contains the requested contents carefully.

* Example of readme.txt:
Runtime per image[s] : 10.43
CPU[0] / GPU[1] : 1
No Extra Data [0] / Extra Data [1] : 0
Data-part among validation [0] or test [1]: 0
Other description: The solution uses the MethodA of Jang et al. ICCV 2022 as a base model. On top of the base model, we applied a novel ray selection method, which helps the proposed neural rendering model initiate the sampling process efficiently and reliably. We have a Python/C++ implementation and report single-core CPU runtime. The method was trained mainly on the Imperial Light-Stage Head dataset, but in the middle of training, we added additional head images from DatasetB, which were introduced in CVPR 2022 by Young et al. to specifically outperform the case of artifactC.

--------------------------------
### Usage of tip1_visualise_camera_poses.py

* Example command: 
> python tip1_visualise_camera_poses.py --in_dir ./ToyEx/000_00 --out_path ./camera_poses.mp4 --save_jpg False

* Arguments explained: 
- in_dir: The path to a subject directory of the ToyEx (e.g., /ToyEx/000_00).
- out_path: The path to the output MP4 file.
- save_jpg: A binary flag ('True' or 'False') to save the output file in JPEG format.

* Description:
This file demonstrates how to convert camera poses from Blender format to OpenCV format. The output file visualizes the camera poses.