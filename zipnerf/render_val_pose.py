import os
import gin
import cv2

from internal import datasets
from internal import configs
from internal import utils
from internal import models
from internal import image
from internal import vis
from internal import checkpoints

import numpy as np
from absl import app

import torch
from torch.utils._pytree import tree_map
import accelerate
from tqdm import tqdm
from pyquaternion import Quaternion

configs.define_common_flags()


def rotation_quaternion(roll, pitch, yaw):
	roll_rad = np.radians(roll)
	pitch_rad = np.radians(pitch)
	yaw_rad = np.radians(yaw)

	q_roll = Quaternion(axis=[1, 0, 0], angle=roll_rad)
	q_pitch = Quaternion(axis=[0, 1, 0], angle=pitch_rad)
	q_yaw = Quaternion(axis=[0, 0, 1], angle=yaw_rad)

	q = q_yaw * q_pitch * q_roll
	return q



def warp_image(image, T, K):
	# Step 2: Get 2D pixel coordinates (homogeneous coordinates)
	h, w = image.shape[:2]
	x, y = np.meshgrid(range(w), range(h))
	P2D = np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))
	
	
	# Step 3: Unproject 2D points to 3D points
	P3D = np.dot(np.linalg.inv(K), P2D)
	
	tmp_rot = np.eye(4)
	tmp_rot[:3,:3] = T
	T = tmp_rot

	# Step 4: Apply the transformation to the 3D points
	P3D_t = np.dot(T, np.vstack((P3D, np.ones_like(P3D[0]))))[:3] / P3D[2]

	# Step 5: Project the transformed 3D points back to 2D image plane
	P2D_t = np.dot(K, P3D_t)

	# Step 6: Find homography matrix
	P2D_t /= P2D_t[2]
	H, _ = cv2.findHomography(P2D[:2].T, P2D_t[:2].T)

	# Step 7: Warp the original image
	warped_image = cv2.warpPerspective(image, H, (w, h))

	return warped_image


def main(unused_argv):
	config = configs.load_config()
	config.exp_path = os.path.join("exp", config.exp_name)
	config.checkpoint_dir = os.path.join(config.exp_path, 'checkpoints')
	with utils.open_file(os.path.join(config.exp_path, 'config.gin'), 'w') as f:
		f.write(gin.config_str())

	# accelerator for DDP
	accelerator = accelerate.Accelerator()

	config.world_size = accelerator.num_processes
	config.global_rank = accelerator.process_index

	# Set random seed.
	accelerate.utils.set_seed(config.seed, device_specific=True)
	model = models.Model(config=config)

	postprocess_fn = lambda z, _=None: z

	# use accelerate to prepare.
	model = accelerator.prepare(model)

	if config.resume_from_checkpoint:
		init_step = checkpoints.restore_checkpoint(config.checkpoint_dir, accelerator)
	else:
		init_step = 0

	os.makedirs(os.path.join(config.data_dir, 'aug_images/'), exist_ok=True)
	aug_idx = len(os.listdir(os.path.join(config.data_dir, 'aug_images/')))

	test_dataset = datasets.load_dataset('val', config.data_dir, config)
	test_dataloader = torch.utils.data.DataLoader(np.arange(len(test_dataset)),
													num_workers=4,
													shuffle=False,
													batch_size=1,
													persistent_workers=True,
													collate_fn=test_dataset.collate_fn,
													)

	# metric handler
	metric_harness = image.MetricHarness()


	for idx, test_batch in enumerate(tqdm(test_dataloader)):
		test_batch = accelerate.utils.send_to_device(test_batch, accelerator.device)
		# render a single image with all distributed processes
		rendering = models.render_image(model, accelerator,
										test_batch, False,
										1, config)
		
		# move to numpy
		rendering = tree_map(lambda x: x.detach().cpu().numpy(), rendering)
		test_batch = tree_map(lambda x: x.detach().cpu().numpy() if x is not None else None, test_batch)

		# Log eval summaries on host 0.
		if accelerator.is_main_process:
			num_rays = np.prod(test_batch['directions'].shape[:-1])

			metric = metric_harness(
				postprocess_fn(rendering['rgb']), postprocess_fn(test_batch['rgb']))

			if config.vis_decimate > 1:
				d = config.vis_decimate
				decimate_fn = lambda x, d=d: None if x is None else x[::d, ::d]
			else:
				decimate_fn = lambda x: x
			rendering = tree_map(decimate_fn, rendering)
			test_batch = tree_map(decimate_fn, test_batch)
			vis_suite = vis.visualize_suite(rendering, test_batch)

			q = rotation_quaternion(config.aug_roll, config.aug_pitch, config.aug_yaw)

			warped_image = warp_image(test_batch['rgb'], np.linalg.inv(q.rotation_matrix), np.linalg.inv(test_dataset.pixtocams))

			cv2.imwrite(f"./render_output/{idx}_test_true_color.png", test_batch['rgb'][...,::-1] * 255)
			cv2.imwrite(f"./render_output/{idx}_test_output_color.png", vis_suite['color'][...,::-1] * 255)
			cv2.imwrite(f"./render_output/{idx}_test_true_warp.png", warped_image[...,::-1] * 255)
			cv2.imwrite(f"{config.data_dir}/aug_images/{aug_idx+idx}.png", warped_image[...,::-1] * 255)
			# with tqdm.external_write_mode():
			# 	print(f'Visualized in {(time.time() - vis_start_time):0.3f}s')
			# tb_process_fn(test_batch['rgb'])
			# summary_writer.add_image('test_true_color', tb_process_fn(test_batch['rgb']), step)
			# if config.compute_normal_metrics:
			# 	summary_writer.add_image('test_true_normals',
			# 							tb_process_fn(test_batch['normals']) / 2. + 0.5, step)
			# for k, v in vis_suite.items():
			# 	summary_writer.add_image('test_output_' + k, tb_process_fn(v), step)
	test_dataset.save_aug_poses(len(os.listdir(os.path.join(config.data_dir, 'aug_images/'))))

if __name__ == '__main__':
	with gin.config_scope('eval'):
		app.run(main)