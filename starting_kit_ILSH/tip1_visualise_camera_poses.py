import numpy as np
from pathlib import Path
# import matplotlib.pyplot as plt
import os
import json

import os, sys
import argparse

import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio

# sys.argv[0]="tip1_visualise_camera_poses.py"
# sys.argv[1:]=["--in_dir", "./ToyEx/000_00",
#               "--out_path", "./camera_poses.mp4",
#               "--save_jpg", "False"]

def load_poses(file_path):
	with open(file_path) as jsonfile:
		data = json.load(jsonfile)
		poses = []
		file_paths = []
		for i, item in enumerate(data['frames']):
			file_path = item['file_path'] + '.jpg'
			pose = np.array(item['transform_matrix'])
			file_paths.append(file_path)
			poses.append(pose)
	poses = np.stack(poses, axis=0)
	
	return poses, file_paths

def get_points_to_visualise_camera_axis (pose_, x_=.1, y_=.1, z_=.1):
	pt_o = np.matmul(pose_, [  0,  0,  0,  1])
	# coordinate of cam origin
	cam_centre_x =  [pt_o[0], pt_o[0], pt_o[0], pt_o[0]]
	cam_centre_y =  [pt_o[1], pt_o[1], pt_o[1], pt_o[1]]
	cam_centre_z =  [pt_o[2], pt_o[2], pt_o[2], pt_o[2]]

	pt_x = np.matmul(pose_, [ x_,  0,  0,  1])
	pt_y = np.matmul(pose_, [  0, y_,  0,  1])
	pt_z = np.matmul(pose_, [  0,  0, z_,  1])
	ray_z = np.matmul(pose_, [  0,  0,  1,  1])
	# coordinate of target points translated along x, y and z axis
	tx =            [pt_x[0], pt_y[0], pt_z[0], ray_z[0]]
	ty =            [pt_x[1], pt_y[1], pt_z[1], ray_z[1]]
	tz =            [pt_x[2], pt_y[2], pt_z[2], ray_z[2]]

	return cam_centre_x, cam_centre_y, cam_centre_z, tx, ty, tz
	
def animate(angle):
	angle *= (36)
	# Normalize the angle to the range [-180, 180] for display
	angle_norm = (angle + 180) % 360 - 180

	# Cycle through a full rotation of elevation, then azimuth, roll, and all
	elev = azim = roll = 0
	if angle <= 360:
		elev = angle_norm
	elif angle <= 360*2:
		azim = angle_norm
	elif angle <= 360*3:
		roll = angle_norm
	else:
		elev = azim = roll = angle_norm

	ax.view_init(elev, azim, roll)
	plt.title('Elevation: %ddegree, Azimuth: %ddegree, Roll: %ddegree' % (elev, azim, roll))
	return line,

def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--in_dir', type=str, help='Path to a subject directory (e.g., /ToyEx/000_00)')
	parser.add_argument('--out_path', type=str, help='Path to the output mp4 file')
	parser.add_argument('--save_jpg', default='False', type=str2bool)
	args = parser.parse_args()
	return args

if __name__=='__main__':
	args = parse_args()
	transforms_train_path = os.path.join(args.in_dir, "transforms_train.json")
	transforms_val_path = os.path.join(args.in_dir, "transforms_val.json")
	poses, file_paths = load_poses(transforms_train_path)
	poses_val, file_paths_val = load_poses(transforms_val_path)

	poses_bounds_aug = np.load(os.path.join(args.in_dir, "poses_bounds_aug.npy"))
	poses_aug = poses_bounds_aug[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
	poses_aug = np.concatenate([poses_aug[..., 1:2], -poses_aug[..., :1], poses_aug[..., 2:4]], -1)

	blender2opencv = np.array([[1,  0,  0,  0], 
							[0, -1,  0,  0],
							[0,  0, -1,  0],
							[0,  0,  0,  1]])

	fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
	
	for idx in range(0, len(file_paths)):
		pose_in_blender = poses[idx]
		pose_in_opencv = np.matmul(pose_in_blender, blender2opencv)
		
		cam_centre_x, cam_centre_y, cam_centre_z, tx, ty, tz \
			= get_points_to_visualise_camera_axis (pose_in_opencv, x_=.1, y_=.1, z_=.1)


		if idx == 0:
			colors=['magenta', 'magenta', 'magenta', 'magenta']
		else:
			colors=['blue', 'green', 'red', 'red']
		
		for i_ in reversed(range(4)):
			if (i_ < 3): mode = 'lines'; opacity=1
			else:  mode = 'lines'; opacity=1

			scatter_3d = go.Scatter3d(x=[cam_centre_x[i_], tx[i_]], 
									y=[cam_centre_y[i_], ty[i_]], 
									z=[cam_centre_z[i_], tz[i_]],
									name=Path(file_paths[idx]).name,
									mode=mode,
									line=dict(color=colors[i_], width=5),
									opacity=opacity)
			fig.add_trace(scatter_3d)

	for idx in range(0, len(file_paths_val)):
		pose_in_blender = poses_val[idx]
		pose_in_opencv = np.matmul(pose_in_blender, blender2opencv)
		
		cam_centre_x, cam_centre_y, cam_centre_z, tx, ty, tz \
			= get_points_to_visualise_camera_axis (pose_in_opencv, x_=.1, y_=.1, z_=.1)

		colors= ['orange', 'orange', 'orange', 'orange']
		for i_ in reversed(range(4)):
			if (i_ < 3): mode = 'lines'; opacity=1
			else:  mode = 'lines'; opacity=1

			scatter_3d = go.Scatter3d(x=[cam_centre_x[i_], tx[i_]], 
									y=[cam_centre_y[i_], ty[i_]], 
									z=[cam_centre_z[i_], tz[i_]],
									name=Path(file_paths[idx]).name,
									mode=mode,
									line=dict(color=colors[i_], width=5),
									opacity=opacity)
			fig.add_trace(scatter_3d)

	for idx in range(0, len(poses_aug)):
		pose_in_blender = poses_aug[idx]
		pose_in_opencv = np.matmul(pose_in_blender, blender2opencv)
		
		cam_centre_x, cam_centre_y, cam_centre_z, tx, ty, tz \
			= get_points_to_visualise_camera_axis (pose_in_opencv, x_=.1, y_=.1, z_=.1)

		colors= ['blue', 'green', 'red', 'cyan']
		for i_ in reversed(range(4)):
			if (i_ < 3): mode = 'lines'; opacity=1
			else:  mode = 'lines'; opacity=1

			scatter_3d = go.Scatter3d(x=[cam_centre_x[i_], tx[i_]], 
									y=[cam_centre_y[i_], ty[i_]], 
									z=[cam_centre_z[i_], tz[i_]],
									name=Path(file_paths[idx]).name,
									mode=mode,
									line=dict(color=colors[i_], width=5),
									opacity=opacity)
			fig.add_trace(scatter_3d)
	
	fig.update_layout(scene=dict(
					xaxis=dict(gridcolor='#F0F0F0',
								backgroundcolor='rgba(0, 0, 255, 0.3)'),
					yaxis=dict(gridcolor='#F0F0F0',
								backgroundcolor='rgba(0, 255, 0, 0.3)'),
					zaxis=dict(gridcolor='#F0F0F0',
								backgroundcolor='rgba(255, 255, 255, 0.3)'),
					bgcolor='rgba(255,255,255,1)'))

	if(args.save_jpg == True):
		fig.update_layout(
			width=1600, height=1600,
			scene=dict(camera=dict(up=dict(x=0, y=0, z=1),
									center=dict(x=0, y=0, z=0),
									eye=dict(x=0.0, y=0.0, z=1),
									projection=dict(type='orthographic'))))
		fig.write_image(args.out_path + '.jpg')
		import cv2
		patch_image = cv2.imread(str(Path(args.in_dir, file_paths[0])))
		patch_image = cv2.resize(patch_image, (300, 400))
		bg_image = cv2.imread(args.out_path + '.jpg')
		bg_image[0:400, 0:300] = patch_image
		cv2.imwrite(args.out_path + '.jpg', bg_image)
		print('Saved file:', args.out_path + '.jpg', '\n')
	else:
		fig.write_html(args.out_path + '.html')
		print('Saved file:', args.out_path + '.html', '\n')