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

def return_3d_mask (mask_p):
    seg_labels = np.load(mask_p)
    #seg_labels[seg_labels==0] = 255
    #seg_labels[seg_labels<250] = 0
    #print(seg_labels.shape)
    img_mask = (seg_labels / 255.).astype(np.uint8)

    #img_mask_3d = np.repeat(img_mask[:, :, np.newaxis], 3, axis=2)
    img_mask_3d = img_mask
    return img_mask_3d
def _open_img(img_p, mask_p):
    img = iio.imread(img_p) / 255.

    if img.shape[2] == 4:
        img = img[:, :, :3]

    if(mask_p):
        img_mask_3d = return_3d_mask (mask_p)
        masking=ma.masked_array(img, mask=img_mask_3d)
        return masking
    else:
        return masking
def output_psnr_mse(img_orig, img_out, mask_use=True):
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

def compute_psnr(GT_path, test_path,  mask_path):
    
        return output_psnr_mse(
            _open_img(GT_path, mask_path),
            _open_img(test_path,  mask_path),
            True
            )
    

GT_path='/content/num18/018_00_01.jpg'
test_path='/content/num18/mixer_4096.png'
mask_path='/content/num18/facemask_18.npy'
psnr_value = compute_psnr( GT_path, test_path,  mask_path)

# 결과 출력
print(f"PSNR: {psnr_value} dB")
