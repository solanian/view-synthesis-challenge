import os, sys
import shutil
import argparse

# sys.argv[0]="1_lightstage_subdataset_restructure.py"
# sys.argv[1:]=["--source_dir", "/home/youngky/dataset/Lightstage_HW/download",
#               "--target_dir", "./devPhase",
#               "phase_in", "devphase"]

def copy_folders(source_dir, target_dir):
    # List all files and directories in the source directory
    for file_or_dir in os.listdir(source_dir): 
        full_path = os.path.join(source_dir, file_or_dir)
        new_target_dir = os.path.join(target_dir, file_or_dir)
        print(full_path)
        
        if os.path.isdir(full_path): # If the item is a directory, recursively call the function to copy its contents
            shutil.copytree(full_path, new_target_dir)
        else:
            shutil.copy(full_path, new_target_dir)

def move_folders(source_dir, target_dir):
    # List all files and directories in the source directory
    for file_or_dir in os.listdir(source_dir):
        full_path = os.path.join(source_dir, file_or_dir)
        new_target_dir = os.path.join(target_dir, file_or_dir)
        print(full_path)

        shutil.move(full_path, new_target_dir)
        
    # Remove the source directory
    os.rmdir(source_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, help='Path to the parent directory <PD> of all the extracted folders <PD>/devPhase_01/, <PD>/devPhase_02/, etc')
    parser.add_argument('--target_dir', type=str, help='Path to the combining destination folder, such as <PD>/devPhase or <PD>/chaPhase')
    parser.add_argument('--phase_in', type=str, help='select between \'devphase\' and \'chaphase\'')
    args = parser.parse_args()
        
    phase_specific_folder_list = {'devphase': ['devPhase_01','devPhase_02'],
                                'chaphase': ['devPhase','chaPhase_01','chaPhase_02']}

    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    for folder_name in phase_specific_folder_list[args.phase_in.lower()]:
        move_folders(os.path.join(args.source_dir,folder_name), args.target_dir)

    print("Data copy and restructuring: Done!")
