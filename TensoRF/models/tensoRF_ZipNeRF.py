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


    def compute_densityfeature(self, means):
        # means.shape = [751904, 6, 3]
        coordinate_plane = torch.stack((means[..., self.matMode[0]], means[..., self.matMode[1]],
                                        means[..., self.matMode[2]])).detach().view(3, -1, 6, 2)
        coordinate_line = torch.stack((means[..., self.vecMode[0]], means[..., self.vecMode[1]],
                                       means[..., self.vecMode[2]]))
        # coordinate_line = [751904,6]
        # coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 6, 2)
        # coordinate_line = [3, 751904,6,2]
        sigma_feature = torch.zeros((means.shape[0], means.shape[1]), device=means.device)
        for idx_plane in range(len(self.density_plane)):  # len = 3
            plane_coef_point = F.grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]],
                                             align_corners=True).view(-1, *means.shape[:2])
            # [1,16,751904,6] --(view)--> [16,751904,6]
            line_coef_point = F.grid_sample(self.density_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *means.shape[:2])
            # [1,16,751904,6] --(view)--> [16,751904,6]
            sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)
            # [751904,6]
        # print(sigma_feature.shape) # torch.Size([751904])
        return sigma_feature

    def compute_appfeature(self, means):

        # plane + line basis
        coordinate_plane = torch.stack((means[..., self.matMode[0]], means[..., self.matMode[1]],
                                        means[..., self.matMode[2]])).detach().view(3, -1, 6, 2)
        coordinate_line = torch.stack((means[..., self.vecMode[0]], means[..., self.vecMode[1]],
                                       means[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1,
                                                                                                                  6, 2)

        plane_coef_point, line_coef_point = [], []
        for idx_plane in range(len(self.app_plane)):
            plane_coef_point.append(F.grid_sample(self.app_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                  align_corners=True).view(-1, *means.shape[:2]))
            line_coef_point.append(F.grid_sample(self.app_line[idx_plane], coordinate_line[[idx_plane]],
                                                 align_corners=True).view(-1, *means.shape[:2]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)
        #print(plane_coef_point.shape) #torch.Size([72, 221184, 6])
        return self.basis_mat((plane_coef_point * line_coef_point).T)

    def forward(self, rays_chunk, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1, train_iter=0,
                origins=None, origin_directions=None, cam_dirs=None, radii=None, rand=True, no_warp=False):

        origins = rays_chunk[:, :3]
        before_viewdirs = rays_chunk[:, 3:6]

        # sample points
        # viewdirs = origin_directions
        viewdirs = before_viewdirs

        xyz_sampled, z_vals, ray_valid = self.sample_ray(origins, viewdirs, is_train=is_train, N_samples=N_samples)
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        ray_valid = ray_valid[:,:-1]
        # print(xyz_sampled.shape) #[4096, 462, 3]

        means, stds = self.cast_rays_for_zip_nerf(z_vals, origins, viewdirs, cam_dirs, radii,
                                                   rand=rand, n=7, m=3, std_scale=0.5)
        means = means.float()
        stds = stds.float()

        sigma = torch.zeros(means.shape[:2], device=means.device)
        rgb = torch.zeros((*means.shape[:2], 3), device=means.device)

        if ray_valid.any():
            sigma_feature = self.compute_densityfeature(means[ray_valid])
            #print(sigma_feature.shape) #torch.Size([751882, 6])
            mlp_res = self.densityModule(means[ray_valid], stds[ray_valid], sigma_feature).squeeze()
            # print(mlp_res.shape)  # torch.Size([751911])
            validsigma = self.feature2density(mlp_res)
            sigma[ray_valid] = validsigma
        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)

        app_mask = weight > self.rayMarch_weight_thres

        if app_mask.any():
            app_features = self.compute_appfeature(means[app_mask])
            valid_rgbs = self.renderModule(means[app_mask], stds[app_mask], app_features)
            # print(app_mask.shape) #torch.Size([4096, 461])
            # print(valid_rgbs.shape) #torch.Size([221184, 3])
            # print(rgb.shape) #torch.Size([4096, 461, 3])

            rgb[app_mask] = valid_rgbs

        vis_res = []
        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if white_bg or (is_train and torch.rand((1,)) < 0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])

        rgb_map = rgb_map.clamp(0, 1)

        with torch.no_grad():
            # print(z_vals.shape) #torch.Size([4096, 462])
            depth_map = torch.sum(weight * z_vals[:,:-1], -1)
            depth_map = depth_map + (1. - acc_map) * rays_chunk[:, -1]

        return rgb_map, depth_map, vis_res  # rgb, sigma, alpha, weight, bg_weight