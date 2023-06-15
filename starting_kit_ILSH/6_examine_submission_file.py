import os, sys
import json

# sys.argv[0]="6_examine_submission_file.py"
# sys.argv[1:]=["--sub_file", "./dummy_submission.zip",
#               "--split_info_file", "./devPhase/challenge_data_split_devPhase.json"]

import zipfile
from PIL import Image
from io import BytesIO

import imageio

def return_the_list_of_required_file_names_in_the_current_phase (split_info_file):
    file_name_list =[]

    with open(split_info_file, 'r') as fp:
        sub_id_with_val_test_ids = json.load(fp)

    key_to_list = ''
    PhaseName = split_info_file.split('/')[-1].split('.')[0].split('_')[-1]
    if(PhaseName == 'devPhase'):
        key_to_list = 'validation'
    elif(PhaseName == 'chaPhase'):
        key_to_list = 'test'
    else:
        print('\nPlease use *_devPhase.json for the Development Phase or *_chaPhase.json for the Challenge Phase to examine the required submission file names.\n')

    if(PhaseName == 'devPhase' or PhaseName == 'chaPhase'):
        type_specified_list = []
        for key_tmp in sub_id_with_val_test_ids.keys():
            for i in range(0, len(sub_id_with_val_test_ids[key_tmp][key_to_list])):
                ## We strongly recommend using the lossless file format (.png) (with the optimize=True option if using Pillow library) when generating results.
                ## Only if the total zip file exceeds 300 MB (the CodaLab submission limit), please use the 'JPEG' format with quality=100 (using the Pillow library) when generating results to preserve as much information as possible.
                sub_name = sub_id_with_val_test_ids[key_tmp][key_to_list][i] #+ '.png' 
                file_name_list.append(sub_name)

    return file_name_list
        
def is_number(s):
    s = s.strip()
    if s.isdigit():
        return "int", int(s)
    else:
        try:
            float(s)
            return "float", float(s)
        except ValueError:
            return "string", ""
        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sub_file', type=str, help='Path to submission zip file submission_tmp.zip')
    parser.add_argument('--split_info_file', type=str, help='Path to split file, such as \{devPhase, chaPhase\}/challenge_data_split_devPhase_\{devPhase, chaPhase\}.json')
    args = parser.parse_args()
    
    ## checking zip file size
    zip_file_size_mb = os.path.getsize(args.sub_file) / (1024 ** 2)
    if zip_file_size_mb >= 299: ## The file size submission limit on CodaLab is 300 MB.
        print(f"Warning: The size of the file is {zip_file_size_mb:.2f} MB and is close to or exceeds the 300 MB limit.")
        print(f"please use the 'JPEG' format with quality=100 (using the Pillow library) when generating results to preserve as much information as possible.")
    else:
        print(f"The size of the file is {zip_file_size_mb:.2f} MB.")
        
    dic_order_of_items_in_readme = {'Runtime':'float', 'CPU':'int', 'Extra':'int', 'validation':'int', 'description':'string'}
    keys_readme = list(dic_order_of_items_in_readme.keys())
    
    file_name_list = return_the_list_of_required_file_names_in_the_current_phase (args.split_info_file)

    expected_wh_res = (3000, 4096)
    expected_img_mode = 'RGB'

    cnt_of_correct_pngs = 0
    isErrorOccuredInPNG = False
    isErrorOccuredInTXT = False
    
    cnt_of_correct_order = 0
    cnt_of_correct_val = 0

    isJPEG_used = False
    isPNG_used = False
    with zipfile.ZipFile(args.sub_file, 'r') as myzip:
        for filename in myzip.namelist():
            ## We strongly recommend using the lossless file format (.png) (with the optimize=True option if using Pillow library) when generating results.
            ## Only if the total zip file exceeds 300 MB (the CodaLab submission limit), please use the 'JPEG' format with quality=100 (using the Pillow library) when generating results to preserve as much information as possible.
            if filename.endswith('.png') or filename.endswith('.jpg'):
                if filename.endswith('.jpg'):
                    isJPEG_used = True
                if filename.endswith('.png'):
                    isPNG_used = True

                if(filename.split('.')[0] in file_name_list):
                    with myzip.open(filename) as myfile:
                        img = Image.open(BytesIO(myfile.read()))
                        # print(f"{filename}: {img.size}, {img.mode}")
                        
                        if(img.size == expected_wh_res and img.mode == expected_img_mode):
                            cnt_of_correct_pngs += 1
                        elif(img.size != expected_wh_res):
                            print('Image Error: The resolution of ' + filename + ' needs to be ' + '3000 for width and 4096 for height.' )
                            isErrorOccuredInPNG = True
                        elif(img.mode != expected_img_mode):
                            print('Image Error: The channel # of ' + filename + ' needs to be RGB (3-channel).' )
                            isErrorOccuredInPNG = True

            elif filename.endswith('.txt'):
                if(filename == 'readme.txt'):
                    with myzip.open(filename) as myfile:
                        contents_in_readme = myfile.read().decode('utf-8').split('\n')

                        row_cnt_to_delet = len(contents_in_readme) - 5
                        while(0 < row_cnt_to_delet):
                            if(contents_in_readme[-1].strip() == ''):
                                contents_in_readme.pop()
                            row_cnt_to_delet -= 1
                        if(5 < len(contents_in_readme)):
                            raise ValueError('The readme.txt file should contain a total of five rows.')

                        for r_i in range(0, len(contents_in_readme)):
                            left_part = contents_in_readme[r_i].split(':')[0]
                            right_part = contents_in_readme[r_i].split(':')[1]

                            readme_key = keys_readme[r_i]
                            readme_val = dic_order_of_items_in_readme[readme_key]
                            if(keys_readme[r_i] in left_part):
                                cnt_of_correct_order += 1
                                
                                type_tmp, val_tmp = is_number(right_part)
                                if(type_tmp == readme_val):
                                    if("int" == type_tmp and (val_tmp < 0 or 1 < val_tmp)):
                                        print ("Value Error: The value of \'"+left_part+ "\' should be 0 or 1.")
                                        isErrorOccuredInTXT = True
                                    else:
                                        cnt_of_correct_val += 1
                else:
                    print('The filename with \'.txt\' extension is not named readme.txt.')

        if(isJPEG_used):
            print(f"- JPEG Warning: Make sure that you are using JPEG lossy image format just because the zip file exceeds 300 MB (the CodaLab submission limit).")
            print(f"-- Save JPEG images with the quality=100 option (using the Pillow library) when generating results to preserve as much information as possible.")
        
        if(isPNG_used and isJPEG_used):
            print('- Image Warning: Please avoid mixing image formats in the zip file.')
            isErrorOccuredInPNG = True

        print('\n-------------------------------------------')
        print('* Summary of examination:\n')

        if(cnt_of_correct_pngs == len(file_name_list)):
            print('- The zip contains the expected number of correct images to be submitted.')
        else:
            missing_png_cnt = len(file_name_list) - cnt_of_correct_pngs
            if(0 < missing_png_cnt):
                print('- The zip file has ' + str(missing_png_cnt) + ' missing images.')
            else:
                print('- The zip file has ' + str(-missing_png_cnt) + ' more images.')

        if(isErrorOccuredInPNG):
            print('-- Either a resolution or channel error occurred. See the details above.\n')
        else:
            if(isJPEG_used):
                print('-- There are no errors in the JPEGs.')
            else:
                print('-- There are no errors in the PNGs.')

        print('-------------------------------------------')
        if(cnt_of_correct_val == len(dic_order_of_items_in_readme)):
            print('- The readme file has the expected values in the corresponding rows.')

        if(cnt_of_correct_order == len(dic_order_of_items_in_readme)):
            print('- The readme file has the expected number of rows in the correct order.')
        else:
            wrong_cnt = len(dic_order_of_items_in_readme) - cnt_of_correct_order
            print('- The readme file has ' + str(wrong_cnt) + ' rows in the incorrect order.')
            print('- The order must follow: Runtime per image[s], CPU[0] / GPU[1], No Extra Data [0] / Extra Data [1], Data-part among validation [0] or test [1], and Other description.')
        
        if(isErrorOccuredInTXT):
            print('-- An error has occurred. See the details above.\n')
        else:
            print('-- There are no errors in the TXTs.\n')
        print('-------------------------------------------')