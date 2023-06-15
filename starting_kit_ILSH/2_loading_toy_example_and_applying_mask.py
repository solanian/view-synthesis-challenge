import os, sys

import torch
import torch.nn.functional as F 

import torchvision.io
import numpy as np

# sys.argv[0]="2_loading_toy_example_and_applying_mask.py"
# sys.argv[1:]=["--eval_ex_imgs_dir_path", "./eval_ex_imgs"]

def psnr_loss (pred, target):
    return 10 * torch.log10(255. ** 2 / torch.mean((pred - target) ** 2))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_ex_imgs_dir_path', type=str, help='Path to the eval_ex_imgs directory of the starting kit')
    args = parser.parse_args()
    
    input_dir = args.eval_ex_imgs_dir_path

    res_dir = os.path.join(input_dir, 'res/')
    ref_dir = os.path.join(input_dir, 'ref/')
    ref_img_folder = 'gt_images'
    ref_border_mask_folder = 'gt_border_masks'

    ref_pngs_ = sorted([p for p in os.listdir(os.path.join(ref_dir,ref_img_folder)) if p.lower().endswith('jpg')])
    res_pngs = sorted([p for p in os.listdir(res_dir) if p.lower().endswith('png')])
    if(len(res_pngs)!=len(ref_pngs_)):
        res_pngs += sorted([p for p in os.listdir(res_dir) if p.lower().endswith('jpg')])
        res_pngs.sort()
    mask_pngs = sorted([p for p in os.listdir(os.path.join(ref_dir,ref_border_mask_folder)) if p.lower().endswith('mask.png')]) ## in case there is images
    ref_pngs = sorted(set(ref_pngs_) - set(mask_pngs))

    if not (len(ref_pngs)==len(res_pngs)):
        raise Exception('Expected %d .png images'%len(ref_pngs))
    
    # Get the file names without extensions
    ref_pngs_wo_file_ex = [os.path.splitext(x)[0] for x in ref_pngs]
    res_pngs_wo_file_ex = [os.path.splitext(x)[0] for x in res_pngs]

    if not (set(res_pngs_wo_file_ex) == set(ref_pngs_wo_file_ex)):
        raise Exception('Full images are expected to be submitted with the correct naming.')

    for (ref_im, res_im) in zip(ref_pngs, res_pngs):
        # read image into tensor
        ref_im_tensor = torchvision.io.read_image(os.path.join(ref_dir,ref_img_folder,ref_im)).to(torch.float32)
        res_im_tensor = torchvision.io.read_image(os.path.join(input_dir,'res',res_im)).to(torch.float32)

        # compute psnr between images without mask
        psnr_wo_mask = psnr_loss(res_im_tensor, ref_im_tensor)

        # load npy (mask) file into numpy array
        border_mask = np.load(os.path.join(ref_dir,ref_border_mask_folder,ref_im[:-4]+'.npy'))
        border_mask[border_mask==0] = False
        border_mask[border_mask!=False] = True
        border_mask = border_mask.astype(np.bool_)
        border_mask_3d = np.repeat(border_mask[:, :], 3, axis=2)

        # convert numpy array to torch tensor
        border_mask_tensor = torch.from_numpy(border_mask_3d)
        border_mask_tensor = border_mask_tensor.permute(2, 0, 1)

        # select elements from target and prediction tensors using the mask
        target_masked = torch.masked_select(ref_im_tensor, border_mask_tensor)
        prediction_masked = torch.masked_select(res_im_tensor, border_mask_tensor)

        # compute psnr between images with mask
        psnr_w_mask = psnr_loss(prediction_masked, target_masked)

        print('PSNR for entire region: ', psnr_wo_mask)
        print('PSNR excluding border area: ', psnr_w_mask)
