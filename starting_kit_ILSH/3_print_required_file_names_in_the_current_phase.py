import os, sys
import json

# sys.argv[0]="3_print_required_file_names_in_the_current_phase.py"
# sys.argv[1:]=["--split_info_file", "./devPhase/challenge_data_split_devPhase.json"]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--split_info_file', type=str, help='Path to split file, such as \{devPhase, chaPhase\}/challenge_data_split_devPhase_\{devPhase, chaPhase\}.json')
    args = parser.parse_args()

    with open(args.split_info_file, 'r') as fp:
        sub_id_with_val_test_ids = json.load(fp)

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

    if(PhaseName == 'devPhase' or PhaseName == 'chaPhase'):
        type_specified_list = []
        for key_tmp in sub_id_with_val_test_ids.keys():
            for i in range(0, len(sub_id_with_val_test_ids[key_tmp][key_to_list])):
                sub_name = sub_id_with_val_test_ids[key_tmp][key_to_list][i] + '.png'
                print(sub_name)
        print('readme.txt\n')
