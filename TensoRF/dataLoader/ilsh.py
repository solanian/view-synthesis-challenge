import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from .ray_utils import *
from .utils import camera_utils


def normalize(v):
	"""Normalize a vector."""
	return v / np.linalg.norm(v)


def average_poses(poses):
	"""
	Calculate the average pose, which is then used to center all poses
	using @center_poses. Its computation is as follows:
	1. Compute the center: the average of pose centers.
	2. Compute the z axis: the normalized average z axis.
	3. Compute axis y': the average y axis.
	4. Compute x' = y' cross product z, then normalize it as the x axis.
	5. Compute the y axis: z cross product x.

	Note that at step 3, we cannot directly use y' as y axis since it's
	not necessarily orthogonal to z axis. We need to pass from x to y.
	Inputs:
		poses: (N_images, 3, 4)
	Outputs:
		pose_avg: (3, 4) the average pose
	"""
	# 1. Compute the center
	center = poses[..., 3].mean(0)  # (3)

	# 2. Compute the z axis
	z = normalize(poses[..., 2].mean(0))  # (3)

	# 3. Compute axis y' (no need to normalize as it's not the final output)
	y_ = poses[..., 1].mean(0)  # (3)

	# 4. Compute the x axis
	x = normalize(np.cross(z, y_))  # (3)

	# 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
	y = np.cross(x, z)  # (3)

	pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

	return pose_avg


def center_poses(poses, blender2opencv):
	"""
	Center the poses so that we can use NDC.
	See https://github.com/bmild/nerf/issues/34
	Inputs:
		poses: (N_images, 3, 4)
	Outputs:
		poses_centered: (N_images, 3, 4) the centered poses
		pose_avg: (3, 4) the average pose
	"""
	poses = poses @ blender2opencv
	pose_avg = average_poses(poses)  # (3, 4)
	pose_avg_homo = np.eye(4)
	pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
	pose_avg_homo = pose_avg_homo
	# by simply adding 0, 0, 0, 1 as the last row
	last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
	poses_homo = \
		np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

	poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
	#     poses_centered = poses_centered  @ blender2opencv
	poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

	return poses_centered, pose_avg_homo


def viewmatrix(z, up, pos):
	vec2 = normalize(z)
	vec1_avg = up
	vec0 = normalize(np.cross(vec1_avg, vec2))
	vec1 = normalize(np.cross(vec2, vec0))
	m = np.eye(4)
	m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
	return m


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
	render_poses = []
	rads = np.array(list(rads) + [1.])

	for theta in np.linspace(0., 2. * np.pi * N_rots, N + 1)[:-1]:
		c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
		z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
		render_poses.append(viewmatrix(z, up, c))
	return render_poses


def get_spiral(c2ws_all, near_fars, rads_scale=1.0, N_views=120):
	# center pose
	c2w = average_poses(c2ws_all)

	# Get average pose
	up = normalize(c2ws_all[:, :3, 1].sum(0))

	# Find a reasonable "focus depth" for this dataset
	dt = 0.75
	close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
	focal = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))

	# Get radii for spiral path
	zdelta = near_fars.min() * .2
	tt = c2ws_all[:, :3, 3]
	rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
	render_poses = render_path_spiral(c2w, up, rads, focal, zdelta, zrate=.5, N=N_views)
	return np.stack(render_poses)


def lift_gaussian(d, t_mean, t_var, r_var, diag):
	"""Lift a Gaussian defined along a ray to 3D coordinates."""
	mean = d[..., None, :] * t_mean[..., None]
	eps = torch.finfo(d.dtype).eps
	# eps = 1e-3
	d_mag_sq = torch.sum(d ** 2, dim=-1, keepdim=True).clamp_min(eps)

	if diag:
		d_outer_diag = d ** 2
		null_outer_diag = 1 - d_outer_diag / d_mag_sq
		t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
		xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
		cov_diag = t_cov_diag + xy_cov_diag
		return mean, cov_diag
	else:
		d_outer = d[..., :, None] * d[..., None, :]
		eye = torch.eye(d.shape[-1], device=d.device)
		null_outer = eye - d[..., :, None] * (d / d_mag_sq)[..., None, :]
		t_cov = t_var[..., None, None] * d_outer[..., None, :, :]
		xy_cov = r_var[..., None, None] * null_outer[..., None, :, :]
		cov = t_cov + xy_cov
		return mean, cov


def construct_ray_warps(fn, t_near, t_far, lam=None):
	"""Construct a bijection between metric distances and normalized distances.

	See the text around Equation 11 in https://arxiv.org/abs/2111.12077 for a
	detailed explanation.

	Args:
		fn: the function to ray distances.
		t_near: a tensor of near-plane distances.
		t_far: a tensor of far-plane distances.
		lam: for lam in Eq(4) in zip-nerf

	Returns:
		t_to_s: a function that maps distances to normalized distances in [0, 1].
		s_to_t: the inverse of t_to_s.
	"""
	if fn is None:
		fn_fwd = lambda x: x
		fn_inv = lambda x: x
	elif fn == 'piecewise':
		# Piecewise spacing combining identity and 1/x functions to allow t_near=0.
		fn_fwd = lambda x: torch.where(x < 1, .5 * x, 1 - .5 / x)
		fn_inv = lambda x: torch.where(x < .5, 2 * x, .5 / (1 - x))
	# elif fn == 'power_transformation':
	#     fn_fwd = lambda x: power_transformation(x * 2, lam=lam)
	#     fn_inv = lambda y: inv_power_transformation(y, lam=lam) / 2
	else:
		inv_mapping = {
			'reciprocal': torch.reciprocal,
			'log': torch.exp,
			'exp': torch.log,
			'sqrt': torch.square,
			'square': torch.sqrt,
		}
		fn_fwd = fn
		fn_inv = inv_mapping[fn.__name__]

	s_near, s_far = [fn_fwd(x) for x in (t_near, t_far)]
	t_to_s = lambda t: (fn_fwd(t) - s_near) / (s_far - s_near)
	s_to_t = lambda s: fn_inv(s * s_far + (1 - s) * s_near)
	return t_to_s, s_to_t


def conical_frustum_to_gaussian(d, t0, t1, base_radius, diag, stable=True):
	"""Approximate a conical frustum as a Gaussian distribution (mean+cov).

	Assumes the ray is originating from the origin, and base_radius is the
	radius at dist=1. Doesn't assume `d` is normalized.

	Args:
		d: the axis of the cone
		t0: the starting distance of the frustum.
		t1: the ending distance of the frustum.
		base_radius: the scale of the radius as a function of distance.
		diag: whether or the Gaussian will be diagonal or full-covariance.
		stable: whether or not to use the stable computation described in
		the paper (setting this to False will cause catastrophic failure).

	Returns:
		a Gaussian (mean and covariance).
	"""
	if stable:
		# Equation 7 in the paper (https://arxiv.org/abs/2103.13415).
		mu = (t0 + t1) / 2  # The average of the two `t` values.
		hw = (t1 - t0) / 2  # The half-width of the two `t` values.
		eps = torch.finfo(d.dtype).eps
		# eps = 1e-3
		t_mean = mu + (2 * mu * hw ** 2) / (3 * mu ** 2 + hw ** 2).clamp_min(eps)
		denom = (3 * mu ** 2 + hw ** 2).clamp_min(eps)
		t_var = (hw ** 2) / 3 - (4 / 15) * hw ** 4 * (12 * mu ** 2 - hw ** 2) / denom ** 2
		r_var = (mu ** 2) / 4 + (5 / 12) * hw ** 2 - (4 / 15) * (hw ** 4) / denom
	else:
		# Equations 37-39 in the paper.
		t_mean = (3 * (t1 ** 4 - t0 ** 4)) / (4 * (t1 ** 3 - t0 ** 3))
		r_var = 3 / 20 * (t1 ** 5 - t0 ** 5) / (t1 ** 3 - t0 ** 3)
		t_mosq = 3 / 5 * (t1 ** 5 - t0 ** 5) / (t1 ** 3 - t0 ** 3)
		t_var = t_mosq - t_mean ** 2
	r_var *= base_radius ** 2
	return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cylinder_to_gaussian(d, t0, t1, radius, diag):
		"""Approximate a cylinder as a Gaussian distribution (mean+cov).

	Assumes the ray is originating from the origin, and radius is the
	radius. Does not renormalize `d`.

	Args:
		d: the axis of the cylinder
		t0: the starting distance of the cylinder.
		t1: the ending distance of the cylinder.
		radius: the radius of the cylinder
		diag: whether or the Gaussian will be diagonal or full-covariance.

	Returns:
		a Gaussian (mean and covariance).
	"""
		t_mean = (t0 + t1) / 2
		r_var = radius ** 2 / 4
		t_var = (t1 - t0) ** 2 / 12
		return lift_gaussian(d, t_mean, t_var, r_var, diag)

def cast_rays(tdist, origins, directions, radii, ray_shape, diag=True):
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
	t0 = tdist[..., :-1]
	t1 = tdist[..., 1:]
	if ray_shape == 'cone':
		gaussian_fn = conical_frustum_to_gaussian
	elif ray_shape == 'cylinder':
		gaussian_fn = cylinder_to_gaussian
	else:
		raise ValueError('ray_shape must be \'cone\' or \'cylinder\'')
	means, covs = gaussian_fn(directions, t0, t1, radii, diag)
	means = means + origins[..., None, :]
	return means, covs


class ILSHDataset(Dataset):
	def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, hold_every=0, bg_remove=True, is_ndc=False, use_aug_pose=False):
		"""
		spheric_poses: whether the images are taken in a spheric inward-facing manner
					default: False (forward-facing)
		val_num: number of val images (used for multigpu training, validate same image for all gpus)
		"""

		self.root_dir = datadir
		self.split = split
		self.hold_every = hold_every
		self.is_stack = is_stack
		self.downsample = downsample
		self.define_transforms()
		self.use_bg_remove = bg_remove
		self.is_ndc = is_ndc
		self.use_aug_pose = use_aug_pose
		self.patch_size = 1
		self.near_far = [3.5, 5.5] # scene bound

		self.blender2opencv = np.eye(4)#np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
		self.read_meta()
		self.white_bg = False

		#         self.near_far = [np.min(self.near_fars[:,0]),np.max(self.near_fars[:,1])]
		

		if self.is_ndc:
			self.near_far = [0, 1.0]

		self.scene_bbox = torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]])
		# self.scene_bbox = torch.tensor([[-1.67, -1.5, -1.0], [1.67, 1.5, 1.0]])
		self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
		self.invradius = 1.0 / (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

	def read_meta(self):
		poses_bounds = np.load(os.path.join(self.root_dir, 'poses_bounds_train.npy'))  # (N_images, 17)
		poses_bounds_val = np.load(os.path.join(self.root_dir, "poses_bounds_val.npy")) # (N_images, 17)
		if self.use_bg_remove:
			self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images_bg_remove/*')))
		else:
			self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images/*')))
		

		# load full resolution image then resize
		if self.split in ['train', 'test']:
			assert len(poses_bounds) == len(self.image_paths), \
				'Mismatch between number of images and number of poses! Please rerun COLMAP!'

		

		poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
		
		poses_val = poses_bounds_val[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
		num_val = len(poses_val)

		# self.near_fars = poses_bounds[:, -2:]  # (N_images, 2)
		self.near_fars = np.array([0.4,2.8])
		hwf = poses[:, :, -1]

		# Step 1: rescale focal length according to training resolution
		H, W, self.focal = poses[0, :, -1]  # original intrinsics, same for all images
		self.img_wh = np.array([int(W / self.downsample), int(H / self.downsample)])
		self.focal = [self.focal * self.img_wh[0] / W, self.focal * self.img_wh[1] / H]

		# Step 2: correct poses
		# Original poses has rotation in form "down right back", change to "right up back"
		# See https://github.com/bmild/nerf/issues/34
		if self.use_aug_pose:
			poses_bounds_aug = np.load(os.path.join(self.root_dir, 'poses_bounds_aug.npy'))  # (N_images, 17)
			self.image_paths_aug = sorted(glob.glob(os.path.join(self.root_dir, 'images_aug/*')))
			self.image_paths.extend(self.image_paths_aug)
			poses_aug = poses_bounds_aug[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
			poses = np.concatenate((poses, poses_aug))
		poses = np.concatenate((poses, poses_val))
		self.poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
		# (N_images, 3, 4) exclude H, W, focal
		# self.poses, self.pose_avg = center_poses(poses, self.blender2opencv)

		# Step 3: correct scale so that the nearest depth is at a little more than 1.0
		# See https://github.com/bmild/nerf/issues/34
		near_original = self.near_fars.min()
		scale_factor = near_original * 0.75  # 0.75 is the default parameter
		# the nearest depth is at 1/0.75=1.33
		self.near_fars /= scale_factor
		self.poses[..., 3] /= scale_factor

		# build rendering path
		N_views, N_rots = 120, 2
		tt = self.poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
		up = normalize(self.poses[:, :3, 1].sum(0))
		rads = np.percentile(np.abs(tt), 90, 0)

		self.render_path = get_spiral(self.poses, self.near_fars, N_views=N_views)

		# distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
		# val_idx = np.argmin(distances_from_center)  # choose val image as the closest to
		# center image

		# ray directions for all pixels, same for all images (same H, W, focal)
		W, H = self.img_wh
		self.directions = get_ray_directions_blender(H, W, self.focal)  # (H, W, 3)

		# average_pose = average_poses(self.poses)
		# dists = np.sum(np.square(average_pose[:3, 3] - self.poses[:, :3, 3]), -1)
		if self.hold_every == 0:
			i_test = np.array([1])
		elif self.hold_every == -1:
			i_test = []
		else:
			i_test = np.arange(0, len(self.poses) - num_val, self.hold_every)  # [np.argmin(dists)]

		if self.split == 'train':
			using_indices = list(set(np.arange(len(self.poses) - num_val)) - set(i_test))
		elif self.split == 'test':
			if self.hold_every == -1:
				using_indices = list(np.arange(len(self.poses))[-num_val:])
			else:
				using_indices = i_test
		print("using_indices:{}".format(using_indices))

		self.all_rays = []
		self.all_rgbs = []
		self.all_means = []
		self.all_covs = []
		for i in tqdm(using_indices, desc="preprocessing for rays"):
			c2w = torch.FloatTensor(self.poses[i])
			if i not in range(len(self.image_paths)):
				img = np.zeros((3, self.img_wh[1], self.img_wh[0]))
			else:
				image_path = self.image_paths[i]
				img = Image.open(image_path).convert('RGB')
				if self.downsample != 1.0:
					img = img.resize(self.img_wh, Image.LANCZOS)
			
			img = self.transform(img)  # (3, h, w)

			img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
			self.all_rgbs += [img]

			rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)

			# gaussian sampling from rays
			near, far = self.near_far
			_, s_to_t = construct_ray_warps(None, near, far)

			init_s_near = 0.
			init_s_far = 1.
			
			sdist = torch.cat([
				torch.full_like(torch.Tensor([near]), init_s_near),
				torch.full_like(torch.Tensor([far]), init_s_far)
			],axis=-1)
			
			# Convert normalized distances to metric distances.
			tdist = s_to_t(sdist)

			pix_x_int, pix_y_int = camera_utils.pixel_coordinates(W, H)
			pixtocam = camera_utils.get_pixtocam(self.focal[0], W, H)
			origins, directions, cam_dirs, radii, imageplane = camera_utils.pixels_to_rays(pix_x_int, pix_y_int, pixtocam, c2w)
			means, covs = cast_rays(
				tdist, 
				origins, 
				directions, 
				radii, 
				ray_shape='cone',
				diag=False)
			
			if means.dim() == 4:
				means = means.squeeze(2)
			if covs.dim() == 5:
				covs = covs.squeeze(2)

			means = means.reshape(-1, 3)
			covs = covs.reshape(-1, 3, 3)

			if self.is_ndc:
				rays_o, rays_d = ndc_rays_blender(H, W, self.focal[0], 1.0, rays_o, rays_d)
			# viewdir = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

			self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
			self.all_means += [means]
			self.all_covs += [covs]

		if not self.is_stack:
			self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
			self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w,3)
			self.all_means = torch.cat(self.all_means, 0)
			self.all_covs = torch.cat(self.all_covs, 0)
		else:
			self.all_rays = torch.stack(self.all_rays, 0)   # (len(self.meta['frames]),h,w, 3)
			self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
			self.all_means = torch.stack(self.all_means, 0)
			self.all_covs = torch.stack(self.all_covs, 0)
		print(f"all_rgbs shape :{self.all_rgbs.shape}, all_rays shape :{self.all_rays.shape}")
		print(f"all_means shape :{self.all_means.shape}, all_covs shape :{self.all_covs.shape}")

		if self.use_bg_remove and self.split=="train" :
			self.mask = torch.where(torch.all(self.all_rgbs != torch.tensor([0., 0., 0.]), dim=1))[0]
			print("mask shape :{}".format(self.mask.shape))

	def define_transforms(self):
		self.transform = T.ToTensor()

	def __len__(self):
		if self.use_bg_remove and self.split=="train" :
			return len(self.mask)
		else:
			return len(self.all_rgbs)
		

	def __getitem__(self, idx):
		if self.use_bg_remove and self.split=="train" :
			idx = self.mask[idx]
		sample = {
			'rays': self.all_rays[idx],
			'rgbs': self.all_rgbs[idx],
			'means': self.all_means[idx],
			'covs': self.all_covs[idx],
		}
		return sample
