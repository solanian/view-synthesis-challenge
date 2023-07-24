# -*- coding: utf-8 -*-
import torch
from models.tensoRF import TensorVMSplit
import math
import torch.nn.functional as F
from gridencoder import GridEncoder
import torch.nn as nn
from models.tensorBase import raw2alpha


class TensorVMSplit_ZipNeRF(TensorVMSplit):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorVMSplit_ZipNeRF, self).__init__(aabb, gridSize, device, **kargs)
        self.rgb_mix = nn.Sequential(
        #   nn.Linear(2, 4),
        #   nn.ReLU(),
        #   nn.Linear(4, 2),
        #   nn.ReLU(),
          nn.Linear(6, 3),
        #   nn.ReLU(),
        ).to(device)

    def cast_rays_for_zip_nerf(self, tdist, origins, directions, cam_dirs, radii, rand=True, n=7, m=3, std_scale=0.5, **kwargs):
        """Cast rays (cone- or cylinder-shaped) and featurize sections of it.

      Args:
        tdist: float array, the "fencepost" distances along the ray.
        origins: float array, the ray origin coordinates.
        directions: float array, the ray direction vectors.
        radii: float array, the radii (base radii for cones) of the rays.
        ray_shape: string, the shape of the ray, must be 'cone' or 'cylinder'.
        diag: boolean, whether or not the covariance matrices should be diagonal.

      Returns:
        a tuple of arrays of means and covariances.
      """
        # print("tdist: ", tdist.shape) # (4096,462)
        t0 = tdist[..., :-1, None]
        t1 = tdist[..., 1:, None]
        radii = radii[..., None]

        t_m = (t0 + t1) / 2
        t_d = (t1 - t0) / 2
        j = torch.arange(6, device=tdist.device)
        t = t0 + t_d / (t_d ** 2 + 3 * t_m ** 2) * (t1 ** 2 + 2 * t_m ** 2 + 3 / 7 ** 0.5 * (2 * j / 5 - 1) * (
                (t_d ** 2 - t_m ** 2) ** 2 + 4 * t_m ** 4).sqrt())
        deg = math.pi / 3 * torch.tensor([0, 2, 4, 3, 5, 1], device=tdist.device, dtype=torch.float)
        deg = torch.broadcast_to(deg, t.shape)
        if rand:
            # randomly rotate and flip
            mask = torch.rand_like(t0[..., 0]) > 0.5
            deg = deg + 2 * math.pi * torch.rand_like(deg[..., 0])[..., None]
            deg = torch.where(mask[..., None], deg, math.pi * 5 / 3 - deg)
        else:
            # rotate 30 degree and flip every other pattern
            mask = torch.arange(t.shape[-2], device=tdist.device) % 2 == 0
            mask = torch.broadcast_to(mask, t.shape[:-1])
            deg = torch.where(mask[..., None], deg, deg + math.pi / 6)
            deg = torch.where(mask[..., None], deg, math.pi * 5 / 3 - deg)
        means = torch.stack([
            radii * t * torch.cos(deg) / 2 ** 0.5,
            radii * t * torch.sin(deg) / 2 ** 0.5,
            t
        ], dim=-1).to("cuda")
        stds = std_scale * radii * t / 2 ** 0.5

        # two basis in parallel to the image plane
        rand_vec = torch.randn_like(cam_dirs)
        ortho1 = F.normalize(torch.cross(cam_dirs, rand_vec, dim=-1), dim=-1)
        ortho2 = F.normalize(torch.cross(cam_dirs, ortho1, dim=-1), dim=-1)

        # just use directions to be the third vector of the orthonormal basis,
        # while the cross section of cone is parallel to the image plane
        basis_matrix = torch.stack([ortho1, ortho2, directions], dim=-1)
        means = torch.matmul(means, basis_matrix[..., None, :, :].transpose(-1, -2).to("cuda"))
        means = means + origins[..., None, None, :]
        # import trimesh
        # trimesh.Trimesh(means.reshape(-1, 3).detach().cpu().numpy()).export("test.ply", "ply")

        return means, stds
    
    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz},
                     {'params': self.density_plane, 'lr': lr_init_spatialxyz},
                     {'params': self.app_line, 'lr': lr_init_spatialxyz},
                     {'params': self.app_plane, 'lr': lr_init_spatialxyz},
                     {'params': self.basis_mat.parameters(), 'lr': lr_init_network}]
        if isinstance(self.renderModule1, torch.nn.Module):
            grad_vars += [{'params': self.renderModule1.parameters(), 'lr': lr_init_network}]
        if isinstance(self.renderModule2, torch.nn.Module):
            grad_vars += [{'params': self.renderModule2.parameters(), 'lr': lr_init_network}]
        return grad_vars

    def forward(self, rays_chunk, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1, train_iter=0,
                origins=None, origin_directions=None, cam_dirs=None, radii=None, rand=True, no_warp=False):

        origins = rays_chunk[:, :3]
        before_viewdirs = rays_chunk[:, 3:6]

        # sample points
        # viewdirs = origin_directions
        viewdirs = before_viewdirs

        xyz_sampled, z_vals, ray_valid = self.sample_ray(origins, viewdirs, is_train=is_train, N_samples=N_samples)
        dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)

        #
        # # print(viewdirs.shape, means.shape, stds.shape) # torch.Size([4096, 3]) torch.Size([4096, 461, 6, 3]) torch.Size([4096, 461, 6])
        # x = self.predict_density(means, stds, rand=rand, no_warp=no_warp)
        # mask_outbbox = ((self.aabb[0] > x) | (x > self.aabb[1])).any(dim=-1)
        # ray_valid = mask_outbbox
        #
        # xyz_sampled = x
        # dists = dists[:,:-1]
        # print(dists.shape)
        # print("$$$$$$$$")
        
        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

        if ray_valid.any():
            xyz_sampled = self.normalize_coord(xyz_sampled)   # 약 -4.5 ~ 4.5 범위로 변경됨
            sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid])

            validsigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma
            
		### zip-nerf encoding - Start ##########################
        means, stds = self.cast_rays_for_zip_nerf(z_vals, origins, viewdirs, cam_dirs, radii,
                                                   rand=rand, n=7, m=3, std_scale=0.5)

        means = torch.cat([means, torch.unsqueeze(means[:, -1,...], 1)], 1).float()
        stds = torch.cat([stds, torch.unsqueeze(stds[:, -1,...], 1)], 1).float()
        
		### zip-nerf encoding - End ############################
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)
        # print("@@@@@")
        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)

        app_mask = weight > self.rayMarch_weight_thres

        if app_mask.any():
            app_features = self.compute_appfeature(xyz_sampled[app_mask])
            if self.shadingMode == 'MLP_freenerf_Fea':
                ### increase the number of positional encoding
                if train_iter > 40000:
                    view_pe, fea_pe = 8, 8
                elif train_iter > 30000:
                    view_pe, fea_pe = 6, 6
                elif train_iter > 20000:
                    view_pe, fea_pe = 4, 4
                elif train_iter > 10000:
                    view_pe, fea_pe = 2, 2
                else:
                    view_pe, fea_pe = 0, 0
                valid_rgbs = self.renderModule(xyz_sampled[app_mask], viewdirs[app_mask], app_features,
                                               v_pe=view_pe, f_pe=fea_pe)
            elif self.shadingMode == 'MLP_ZipNeRF_Fea':
                valid_rgbs = self.renderModule(means[app_mask], stds[app_mask], app_features)
            elif self.shadingMode == 'mixed':
                valid_rgbs1 = self.renderModule1(means[app_mask], stds[app_mask], app_features)
                ### increase the number of positional encoding
                if train_iter > 40000:
                    view_pe, fea_pe = 8, 8
                elif train_iter > 30000:
                    view_pe, fea_pe = 6, 6
                elif train_iter > 20000:
                    view_pe, fea_pe = 4, 4
                elif train_iter > 10000:
                    view_pe, fea_pe = 2, 2
                else:
                    view_pe, fea_pe = 0, 0
                valid_rgbs2 = self.renderModule2(xyz_sampled[app_mask], viewdirs[app_mask], app_features,
                                               v_pe=view_pe, f_pe=fea_pe)
                valid_rgbs = valid_rgbs1 + valid_rgbs2
                # valid_rgbs = torch.stack((valid_rgbs1, valid_rgbs2), dim=-1)
                # valid_rgbs = self.rgb_mix(valid_rgbs).squeeze(-1)
            else:
                valid_rgbs = self.renderModule(xyz_sampled[app_mask], viewdirs[app_mask], app_features)
            rgb[app_mask] = valid_rgbs

        vis_res = [xyz_sampled[app_mask], viewdirs[app_mask], rgb[app_mask]]
        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if white_bg or (is_train and torch.rand((1,)) < 0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])

        rgb_map = rgb_map.clamp(0, 1)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)
            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]

        return rgb_map, depth_map, vis_res  # rgb, sigma, alpha, weight, bg_weight