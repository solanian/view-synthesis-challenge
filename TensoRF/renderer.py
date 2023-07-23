import torch,os,imageio,sys
from tqdm.auto import tqdm
from dataLoader.ray_utils import get_rays
from utils import *
from dataLoader.ray_utils import ndc_rays_blender
import csv


def zipnerf_renderer(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False,
                                img_wh=[0, 0], train_iter=0, device='cuda', origins=None, origin_directions=None,
                              cam_dirs=None, radiis=None ):
    rgbs, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
    N_rays_all = origins.shape[0]

    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        origins_chunk = origins[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        origin_directions_chunk = origin_directions[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        cam_dirs_chunk = cam_dirs[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        radii_chunk = radiis[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)

        rgb_map, depth_map, vis_res = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray,
                                                     N_samples=N_samples, train_iter=train_iter,
                                                     origins=origins_chunk, origin_directions=origin_directions_chunk,
                                                     cam_dirs=cam_dirs_chunk, radii=radii_chunk)
        rgbs.append(rgb_map)
        depth_maps.append(depth_map)

    return torch.cat(rgbs), None, torch.cat(depth_maps), None, None

def OctreeRender_trilinear_fast(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False,
                                img_wh=[0,0], train_iter=0, device='cuda'):

    rgbs, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
    N_rays_all = rays.shape[0]

    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        rgb_map, depth_map, vis_res = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray,
                                              N_samples=N_samples, train_iter=train_iter)

        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
    
    return torch.cat(rgbs), None, torch.cat(depth_maps), None, None


def OctreeRender_one_cam_3d_vis(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False,
                                    img_wh=(0,0), train_iter=50000000, device='cuda'):
    rgbs, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
    N_rays_all = rays.shape[0]


    print("OctreeRender_one_cam_3d_vis ray_shape : {}".format(rays.shape))
    points_cloud = []
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        rgb_map, depth_map, vis_res = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray,
                                              N_samples=N_samples, train_iter=train_iter)

        rgbs.append(rgb_map)
        depth_maps.append(depth_map)

        # draw points over the weight threshold
        xyz, viewdirs, rgb = vis_res
        for idx in list(range(0, chunk, 1024)):
            points_cloud.append([float(xyz[idx][0]), float(xyz[idx][1]), float(xyz[idx][2]), 1,
                              float(rgb[idx][0]), float(rgb[idx][1]), float(rgb[idx][2]), "samples"])

    # draw camera position
    ray_o = rays[0,:3].tolist()
    points_cloud.append([ray_o[0], ray_o[1], ray_o[2], 5, 1.0, 0, 0, "cam_loc"])

    # draw frustum edges
    near_far = tensorf.near_far
    for i in [0, img_wh[0]-1, N_rays_all-img_wh[0], N_rays_all-1]:
        d = rays[i, 3:6].tolist()
        for j in [near_far[0], near_far[1]]:
            points_cloud.append([ray_o[0]+d[0]*j, ray_o[1]+d[1]*j, ray_o[2]+d[2]*j, 1, 0, 1.0, 0, "view_dir"])

    csv_file_path = "exp/one_cam_points_cloud.csv"
    f = open(csv_file_path, 'a')
    writer = csv.writer(f)
    writer.writerows(points_cloud)
    f.close()

    return torch.cat(rgbs), None, torch.cat(depth_maps), None, None

def OctreeRender_multi_cam_3d_vis(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=True,
                                    img_wh=(0,0), train_iter=50000000, device='cuda'): #is_train option is TRUE!!
    rgbs, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
    N_rays_all = rays.shape[0]


    print("OctreeRender_multi_cam_3d_vis ray_shape : {}".format(rays.shape))
    points_cloud = []
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        rgb_map, depth_map, vis_res = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray,
                                              N_samples=N_samples, train_iter=train_iter)

        rgbs.append(rgb_map)
        depth_maps.append(depth_map)

        # draw points over the weight threshold
        xyz, viewdirs, rgb = vis_res
        for idx in list(range(0, chunk, 2047)):
            points_cloud.append([float(xyz[idx][0]), float(xyz[idx][1]), float(xyz[idx][2]), 1,
                              float(rgb[idx][0]), float(rgb[idx][1]), float(rgb[idx][2]), "samples"])

    # draw camera position
    ray_o = rays[0,:3].tolist()
    points_cloud.append([ray_o[0], ray_o[1], ray_o[2], 10.0, 1.0, 0, 0, "cam_loc"])

    # draw frustum edges
    near_far = tensorf.near_far
    for i in [0, img_wh[0]-1, N_rays_all-img_wh[0], N_rays_all-1]:
        d = rays[i, 3:6].tolist()
        for j in [near_far[0], near_far[1]]:
            points_cloud.append([ray_o[0]+d[0]*j, ray_o[1]+d[1]*j, ray_o[2]+d[2]*j, 1, 0, 1.0, 0, "view_dir"])

    csv_file_path = "exp/multi_cam_points_cloud.csv"
    f = open(csv_file_path, 'a')
    writer = csv.writer(f)
    writer.writerows(points_cloud)
    f.close()

    return torch.cat(rgbs), None, torch.cat(depth_maps), None, None


@torch.no_grad()
def evaluation(test_dataset,tensorf, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=False, device='cuda', train_iter=5000000):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    all_origins = test_dataset.all_origins
    all_directions = test_dataset.all_directions
    all_cam_dirs = test_dataset.all_cam_dirs
    all_radii = test_dataset.all_radii


    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis,1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
    for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):

        W, H = test_dataset.img_wh
        rays = samples.view(-1,samples.shape[-1])
        origins = torch.squeeze(all_origins,0) ## <- this code is for only one test image. it should be fixed!!!
        origin_directions = torch.squeeze(all_directions,0) ## same
        cam_dirs = torch.squeeze(all_cam_dirs,0) ## same
        radii = torch.squeeze(all_radii,0) ## same


        if args.model_name == "TensorVMSplit_ZipNeRF":
            rgb_map, alphas_map, depth_map, weights, uncertainty = zipnerf_renderer(rays, tensorf,
                                                                                        chunk=4096,
                                                                                        N_samples=N_samples,
                                                                                        ndc_ray=ndc_ray,
                                                                                        white_bg=white_bg,
                                                                                        is_train=False,
                                                                                        img_wh=[0,0],
                                                                                        train_iter=train_iter,
                                                                                        device=device,
                                                                                        origins=origins,
                                                                                        origin_directions=origin_directions,
                                                                                        cam_dirs=cam_dirs,
                                                                                        radiis=radii)
        else:
            rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(rays, tensorf, chunk=4096,
                                    N_samples=N_samples, white_bg = white_bg, ndc_ray=ndc_ray, device=device, train_iter=train_iter)

        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensorf.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensorf.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    # imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
    # imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))

    return PSNRs

@torch.no_grad()
def evaluation_path(test_dataset,tensorf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    for idx, c2w in tqdm(enumerate(c2ws)):

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=8192, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=8)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=8)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs
