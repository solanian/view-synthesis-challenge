#!/usr/bin/env python
import time

import os.path
import os, sys
import numpy as np

# import imageio
import imageio.v3 as iio

import numpy.ma as ma
import warnings

from multiprocessing import Pool
from skimage.metrics import structural_similarity as rgb_ssim
from math import ceil

# sys.argv[0]="5_evaluation_script_example.py"
# sys.argv[1:]=["--eval_ex_imgs_dir_path", "./eval_ex_imgs"]

def return_3d_mask (mask_p):
    seg_labels = np.load(mask_p)
    seg_labels[seg_labels==0] = 255
    seg_labels[seg_labels<250] = 0
    
    img_mask = (seg_labels / 255.).astype(np.uint8)

    img_mask_3d = np.repeat(img_mask[:, :, np.newaxis], 3, axis=2)

    return img_mask_3d

def output_psnr_mse(img_orig, img_out, mask_use=False):
    unique_values = np.unique(img_out)
    if((mask_use and len(unique_values) == 2) or
       (not mask_use and len(unique_values) == 1)): ## When images that are entirely black or white are submitted, it returns 0 without performing any calculations.
            return 0
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            psnr = -10. * np.log10(np.mean(np.square(img_orig - img_out)))
    except RuntimeWarning:
        # psnr = float('inf')
        psnr = 0

    return psnr

def output_ssim_mse(img_orig, img_out, mask_use=False, mask_p=None):
    unique_values = np.unique(img_out)
    if((mask_use and len(unique_values) == 2) or
       (not mask_use and len(unique_values) == 1)): ## When images that are entirely black or white are submitted, it returns 0 without performing any calculations.
            return 0, 0
    
    ssim, ssim_img = rgb_ssim(img_orig.astype(np.float32), img_out.astype(np.float32),
                  data_range=img_orig.max() - img_orig.min(), channel_axis=-1, full=True)

    ssim_masked = None
    if(mask_use):
        bool_mask = return_3d_mask (mask_p).astype(bool)
        ssim_masked = np.mean(ssim_img[~bool_mask])

    return ssim, ssim_masked

def _open_img(img_p, mask_p):
    img = iio.imread(img_p) / 255.
    
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if(mask_p):
        img_mask_3d = return_3d_mask (mask_p)
        
        return ma.masked_array(img, mask=img_mask_3d)
    else:
        return img

def compute_psnr(ref_im, res_im, mask_p=None):
    if(mask_p!=None):
        return output_psnr_mse(
            _open_img(os.path.join(ref_dir,ref_img_folder,ref_im), os.path.join(ref_dir,ref_mask_folder,ref_im[:-4]+mask_p+'.npy')),
            _open_img(os.path.join(res_dir,res_im), os.path.join(ref_dir,ref_mask_folder,ref_im[:-4]+mask_p+'.npy')),
            True
            )
    else:
        return output_psnr_mse(
            _open_img(os.path.join(ref_dir,ref_img_folder,ref_im), mask_p),
            _open_img(os.path.join(input_dir,'res',res_im), mask_p),
            False
            )


def compute_ssim(ref_im, res_im, mask_p=None):
    if(mask_p!=None):
        return output_ssim_mse(
            _open_img(os.path.join(ref_dir,ref_img_folder,ref_im), mask_p),
            _open_img(os.path.join(input_dir,'res',res_im), mask_p),
            True, 
            os.path.join(ref_dir,ref_mask_folder,ref_im[:-4]+mask_p+'.npy')
            )
    else:
        return output_ssim_mse(
            _open_img(os.path.join(ref_dir,ref_img_folder,ref_im), mask_p),
            _open_img(os.path.join(input_dir,'res',res_im), mask_p),
            False
            )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_ex_imgs_dir_path', type=str, help='Path to the eval_ex_imgs directory of the starting kit')
    args = parser.parse_args()

    input_dir = args.eval_ex_imgs_dir_path
    output_dir = input_dir

    res_dir = os.path.join(input_dir, 'res/')
    ref_dir = os.path.join(input_dir, 'ref/')
    ref_img_folder = 'gt_images'
    ref_mask_folder = 'gt_face_masks'

    subset_feedback_cnt = 3


    runtime = -1
    cpu = -1
    data = -1
    phase_ = -1
    other = ""
    readme_fnames = [p for p in os.listdir(res_dir) if p.lower().startswith('readme')]
    try:
        readme_fname = readme_fnames[0]
        print("Parsing extra information from %s"%readme_fname)
        with open(os.path.join(input_dir, 'res', readme_fname)) as readme_file:
            readme = readme_file.readlines()
            lines = [l.strip() for l in readme if l.find(":")>=0]
            runtime = float(":".join(lines[0].split(":")[1:]))
            cpu = int(":".join(lines[1].split(":")[1:]))
            data = int(":".join(lines[2].split(":")[1:]))
            phase_ = int(":".join(lines[3].split(":")[1:]))
            other = ":".join(lines[4].split(":")[1:])
    except:
        print("Error occured while parsing readme.txt")
        print("Please make sure you have a line for runtime, cpu/gpu, extra data and other (4 lines in total).")
    # print("Parsed information:")
    # print("Runtime/Img: %f"%runtime)
    # print("CPU/GPU: %d"%cpu)
    # print("ExtraData: %d"%data)
    # print("Val/Test: %d"%phase_)
    # print("OtherDesc: %s"%other)

    ref_pngs_ = sorted([p for p in os.listdir(os.path.join(ref_dir,ref_img_folder)) if p.lower().endswith('jpg')])
    res_pngs = sorted([p for p in os.listdir(res_dir) if p.lower().endswith('png')])
    if(len(res_pngs)!=len(ref_pngs_)):
        res_pngs += sorted([p for p in os.listdir(res_dir) if p.lower().endswith('jpg')])
        res_pngs.sort()
    mask_pngs = sorted([p for p in os.listdir(os.path.join(ref_dir,ref_mask_folder)) if p.lower().endswith('mask.png')]) ## in case there is images
    ref_pngs = sorted(set(ref_pngs_) - set(mask_pngs))

    if not (len(ref_pngs)==len(res_pngs)):
        raise Exception('Expected %d .png images'%len(ref_pngs))
    
    # Get the file names without extensions
    ref_pngs_wo_file_ex = [os.path.splitext(x)[0] for x in ref_pngs]
    res_pngs_wo_file_ex = [os.path.splitext(x)[0] for x in res_pngs]

    if not (set(res_pngs_wo_file_ex) == set(ref_pngs_wo_file_ex)):
        raise Exception('Full images are expected to be submitted with the correct naming.')

    def compute_psnr_multi(pair):
        ref_im, res_im, mask_p = pair
        return [ref_im, compute_psnr(ref_im, res_im, mask_p)]

    def compute_ssim_multi(pair):
        ref_im, res_im, mask_p = pair
        ssim, ssim_masked = compute_ssim(ref_im, res_im, mask_p)
        return [ref_im, ssim, ssim_masked]

    psnrs = []
    psnrs_masked = []
    
    with Pool() as pool:
        psnrs = pool.map(compute_psnr_multi, zip(ref_pngs, res_pngs, [None for _ in ref_pngs]))
        psnrs_masked = pool.map(compute_psnr_multi, zip(ref_pngs, res_pngs, ['' for _ in ref_pngs]))

    # sort the outer list based on the first element of each sublist
    psnrs.sort(key=lambda x: x[0])
    psnrs_list = [x[1] for x in psnrs]
    psnrs_masked.sort(key=lambda x: x[0])
    psnrs_masked_list = [x[1] for x in psnrs_masked]

    psnr = np.mean(psnrs_list)
    psnr_masked = np.mean(psnrs_masked_list)

    ssims = []
    ssims_masked = []

    n_tasks = len(ref_pngs)
    n_tasks_per_chunk = ceil(n_tasks / len(pool._pool))

    with Pool() as pool:
        ssims_w_wo_masked = pool.map(compute_ssim_multi, zip(ref_pngs, res_pngs, ['' for _ in ref_pngs]), chunksize=n_tasks_per_chunk)
    ssims = [[x[0], x[1]] for x in ssims_w_wo_masked]
    ssims_masked = [[x[0], x[2]] for x in ssims_w_wo_masked]

    # sort the outer list based on the first element of each sublist
    ssims.sort(key=lambda x: x[0])
    ssims_list = [x[1] for x in ssims]
    ssims_masked.sort(key=lambda x: x[0])
    ssims_masked_list = [x[1] for x in ssims_masked]

    ssim = np.mean(ssims_list)
    ssim_masked = np.mean(ssims_masked_list)

    sub_psnr = np.mean(psnrs_list[:subset_feedback_cnt])
    sub_psnr_masked = np.mean(psnrs_masked_list[:subset_feedback_cnt])
    sub_ssim = np.mean(ssims_list[:subset_feedback_cnt])
    sub_ssim_masked = np.mean(ssims_masked_list[:subset_feedback_cnt])

    with open(os.path.join(output_dir, 'scores.txt'), 'w') as output_file:
        output_file.write("Full_data_PSNR:%f\n"%psnr)
        output_file.write("Full_data_PSNR_masked:%f\n"%psnr_masked)
        output_file.write("Full_data_SSIM:%f\n"%ssim)
        output_file.write("Full_data_SSIM_masked:%f\n"%ssim_masked)
        output_file.write("Three_imgs_PSNR:%f\n"%sub_psnr)
        output_file.write("Three_imgs_PSNR_masked:%f\n"%sub_psnr_masked)
        output_file.write("Three_imgs_SSIM:%f\n"%sub_ssim)
        output_file.write("Three_imgs_SSIM_masked:%f\n"%sub_ssim_masked)
        output_file.write("ExtraRuntime:%f\n"%runtime)
        output_file.write("ExtraPlatform:%d\n"%cpu)
        output_file.write("ExtraData:%d\n"%data)
        output_file.write("ExtraPhase:%d\n"%phase_)
        output_file.write("ExtraOtherDesc:%s\n"%other)
        output_file.write("set1_score:%f"%psnr_masked)
        
    with open(os.path.join(output_dir, 'scores.txt'), 'r') as f:
        contents = f.read()
        print(contents)