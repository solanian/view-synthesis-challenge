import os, sys
import json
from PIL import Image
import zipfile

# sys.argv[0]="4_create_dummy_submission_zip.py"
# sys.argv[1:]=["--split_info_file", "./devPhase/challenge_data_split_devPhase.json",
#               "--target_dir", "./dummyDir"]

def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            ziph.write(file_path, os.path.basename(file_path))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--split_info_file', type=str, help='Path to split file, such as \{devPhase, chaPhase\}/challenge_data_split_devPhase_\{devPhase, chaPhase\}.json')
    parser.add_argument('--target_dir', type=str, help='Path to the dummy folder, which is created to demonstrate the collection of dummy submission files')
    
    args = parser.parse_args()

    with open(args.split_info_file, 'r') as fp:
        sub_id_with_val_test_ids = json.load(fp)

    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    key_to_list = ''
    PhaseName = args.split_info_file.split('/')[-1].split('.')[0].split('_')[-1]
    if(PhaseName == 'devPhase'):
        key_to_list = 'validation'
        print('\nFor the Developement Phase, please submit a zip file containing the following files:')
    elif(PhaseName == 'chaPhase'):
        key_to_list = 'test'
        print('\nFor the Challenge Phase, please submit a zip file containing the following files:')
    else:
        print('\nPlease use *_devPhase.json for the Development Phase or *_chaPhase.json for the Challenge Phase to list the required submission file names.\n')

    width, height = 3000, 4096
    if(PhaseName == 'devPhase' or PhaseName == 'chaPhase'):
        type_specified_list = []
        for key_tmp in sub_id_with_val_test_ids.keys():
            for i in range(0, len(sub_id_with_val_test_ids[key_tmp][key_to_list])):
                sub_name = sub_id_with_val_test_ids[key_tmp][key_to_list][i] + '.png'
                print(sub_name)
                
                image = Image.new('RGB', (width, height), 'black')
                image.save(os.path.join(args.target_dir, sub_name), optimize=True) ## We suggest using the optimize=True option to reduce file size while still maintaining lossless compression.

        with open(os.path.join(args.target_dir,'readme.txt'), 'w') as f:
            f.write('Runtime per image[s]: 10.43\n')
            f.write('CPU[0] / GPU[1]: 1\n')
            f.write('No Extra Data [0] / Extra Data [1]: 0\n')
            f.write('Data-part among validation [0] or test [1]: 0\n')
            f.write('Other description: The solution uses the MethodA of Jang et al. ICCV 2022 as a base model. On top of the base model, we applied a novel ray selection method, which helps the proposed neural rendering model initiate the sampling process efficiently and reliably. We have a Python/C++ implementation and report single-core CPU runtime. The method was trained mainly on the Imperial Light-Stage Head dataset, but in the middle of training, we added additional head images from DatasetB, which were introduced in CVPR 2022 by Young et al. to specifically outperform the case of artifactC.')
        
        zip_file = zipfile.ZipFile(os.path.join(args.target_dir,'../dummy_submission.zip'), 'w', zipfile.ZIP_DEFLATED)
        zipdir(args.target_dir, zip_file)
        zip_file.close()
