import numpy as np
from pathlib import Path
# import matplotlib.pyplot as plt
import os
import json
import glob
import cv2
from math import *
import os, sys
import argparse
import random
from scipy.spatial.transform import Rotation as R
from rich import print
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio

# sys.argv[0]="tip1_visualise_camera_poses.py"
# sys.argv[1:]=["--in_dir", "./ToyEx/000_00",
#               "--out_path", "./camera_poses.mp4",
#               "--save_jpg", "False"]

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

def create_projection_matrix(intrinsics, pose):
	T = np.eye(4)
	T[:3, :3] = pose[:3, :3]
	T[:3, 3] = pose[:3, 3]

	return intrinsics.dot(T[:3, [0, 1, 3]])

def warp_image(img, source_pose, target_pose, intrinsics):
	height, width, _ = img.shape

	# Create source and target projection matrices
	source_P = create_projection_matrix(intrinsics, source_pose)
	target_P = create_projection_matrix(intrinsics, target_pose)

	# Compute the inverse of the source projection matrix
	source_P_inv = np.linalg.pinv(source_P)

	# Create an empty (black) output image
	warped_img = np.zeros_like(img)

	for x in range(width):
		for y in range(height):
			# Project the pixel coordinate (x, y) to 3D space by raytracing
			pixel_source = np.array([x, y, 1])
			ray_source = np.dot(source_P_inv, pixel_source)

			# Project this 3D coordinate onto the target image
			pixel_target = np.dot(target_P, ray_source)
			pixel_target = (pixel_target[:2] / pixel_target[2]).astype(int)

			# Check if the point lies within the image boundaries
			if 0 <= pixel_target[0] < width and 0 <= pixel_target[1] < height:
				warped_img[pixel_target[1], pixel_target[0]] = img[y, x]

	return warped_img

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

def random_camera_pose(elevation_range, distance_range):
	elevation_min, elevation_max = elevation_range
	distance_min, distance_max = distance_range

	# Generate random angles and distance within the given range
	elevation = random.uniform(elevation_min, elevation_max)
	azimuth = random.uniform(0, 360)
	distance = random.uniform(distance_min, distance_max)

	# Convert spherical coordinates to Cartesian coordinates
	theta, phi = elevation, np.radians(azimuth)
	z = distance * np.cos(phi)
	y = z * np.tan(theta)
	x = z * np.tan(phi)

	# Calculate lookAtDir, upVec, and leftVec
	camera_pos = np.array([x, y, z])
	look_at_pos = np.array([0, 0, 0])
	
	look_at_dir = -(look_at_pos - camera_pos)/np.linalg.norm(look_at_pos - camera_pos)
	up_vec = np.array([0, 1, 0])
	left_vec = np.cross(up_vec, look_at_dir)
	left_vec /= np.linalg.norm(left_vec)
	
	# Create a 90-degree clockwise rotation matrix around the look_at_dir
	rotation_angle = np.radians(-90)
	rotation_matrix = np.array([[np.cos(rotation_angle) + look_at_dir[0] ** 2 * (1 - np.cos(rotation_angle)),
								look_at_dir[0] * look_at_dir[1] * (1 - np.cos(rotation_angle)) -
								look_at_dir[2] * np.sin(rotation_angle),
								look_at_dir[0] * look_at_dir[2] * (1 - np.cos(rotation_angle)) +
								look_at_dir[1] * np.sin(rotation_angle)],

								[look_at_dir[1] * look_at_dir[0] * (1 - np.cos(rotation_angle)) +
								look_at_dir[2] * np.sin(rotation_angle),
								np.cos(rotation_angle) + look_at_dir[1] ** 2 * (1 - np.cos(rotation_angle)),
								look_at_dir[1] * look_at_dir[2] * (1 - np.cos(rotation_angle)) -
								look_at_dir[0] * np.sin(rotation_angle)],

								[look_at_dir[2] * look_at_dir[0] * (1 - np.cos(rotation_angle)) -
								look_at_dir[1] * np.sin(rotation_angle),
								look_at_dir[2] * look_at_dir[1] * (1 - np.cos(rotation_angle)) +
								look_at_dir[0] * np.sin(rotation_angle),
								np.cos(rotation_angle) + look_at_dir[2] ** 2 * (1 - np.cos(rotation_angle))]])

	# Apply the rotation matrix
	left_vec = np.dot(rotation_matrix, left_vec)
	up_vec = np.dot(rotation_matrix, up_vec)
	
	# Assemble the 3x4 transformation matrix
	pose_matrix = np.zeros((3, 4))
	pose_matrix[:, 0] = left_vec
	pose_matrix[:, 1] = up_vec
	pose_matrix[:, 2] = look_at_dir
	pose_matrix[:, 3] = camera_pos

	return pose_matrix


def compute_homography_matrix(pose1, intrinsics1, pose2, intrinsics2):
	# Compute R1, R2, and t1, t2
	R1, R2 = pose1[:3, :3], pose2[:3, :3]
	t1, t2 = pose1[:3, 3], pose2[:3, 3]

	# Compute R, t: relative rotation and translation from pose1 to pose2
	R = np.dot(R2, R1.T)
	t = np.dot(-R2, R1.T).dot(t1) + t2

	# Compute the homography matrix
	H = np.dot(intrinsics2, R - np.outer(t, np.linalg.inv(intrinsics1).dot(np.array([0, 0, 1])))).dot(np.linalg.inv(intrinsics1))
	return H

def warp_image_with_homography(img, homography_matrix):
	height, width, _ = img.shape

	# Warp the image using the homography matrix
	warped_img = cv2.warpPerspective(img, homography_matrix, (width, height))
	return warped_img


def intrinsic_matrix(fx, fy, cx, cy):
	"""Intrinsic matrix for a pinhole camera in OpenCV coordinate system."""
	return np.array([
		[fx, 0, cx],
		[0, fy, cy],
		[0, 0, 1.],
	])


def get_pixtocam(focal, width, height):
	"""Inverse intrinsic matrix for a perfect pinhole camera."""
	camtopix = intrinsic_matrix(focal, focal, width * .5, height * .5)
	return np.linalg.inv(camtopix)

def angle_between_points(p1, p2, origin=np.array([0, 0, 0])):
	# Create vectors from the origin to the points
	v1 = p1 - origin
	v2 = p2 - origin

	# Normalize the vectors
	v1_norm = v1 / np.linalg.norm(v1)
	v2_norm = v2 / np.linalg.norm(v2)

	# Compute the dot product between the two normalized vectors
	dot_product = np.dot(v1_norm, v2_norm)

	# Calculate the angle between the two vectors using the acos function
	angle_rad = np.arccos(np.clip(dot_product, -1, 1))  # Clip to handle floating-point inaccuracies

	return angle_rad

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--in_dir', type=str, help='Path to a subject directory (e.g., /ToyEx/000_00)')
	parser.add_argument('--out_path', type=str, help='Path to the output mp4 file')
	args = parser.parse_args()
	return args

if __name__=='__main__':
	args = parse_args()
	poses_bounds = np.load(os.path.join(args.in_dir, "poses_bounds_train.npy"))
	image_paths = sorted(glob.glob(os.path.join(args.in_dir, 'images_bg_remove/*')))

	poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
	near_fars = poses_bounds[:, -2:]  # (N_images, 2)
	H, W, focal = poses[0, :, -1]
	pixtocams = get_pixtocam(focal, W, H)
	poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)

	pose_dicts = []
	min_dist = min_elev = sys.maxsize
	max_dist = max_elev = 0
	
	for i, pose in enumerate(poses):
		tr = pose[:, 3]
		rot = pose[:, :3]
		dist = np.linalg.norm(tr)
		min_dist = min(min_dist, dist)
		max_dist = max(max_dist, dist)
		rot_R = R.from_matrix(rot)
		elev = asin(tr[1]/dist)
		min_elev = min(min_elev, elev)
		max_elev = max(max_elev, elev)
		pose_dict = dict(
			pose = pose,
			image = cv2.imread(image_paths[i])
		)
		pose_dicts.append(pose_dict)
	
	aug_poses = []

	for i in range(20):
		aug_pose = random_camera_pose([min_elev, max_elev], [min_dist, max_dist])
		aug_poses.append(np.append(np.column_stack((aug_pose, np.array([H, W, focal]))).reshape(-1), near_fars[0]))
		min_ang_dist = sys.maxsize
		min_idx = 0
		for ii, pose_dict in enumerate(pose_dicts):
			ang_dist = angle_between_points(pose_dict['pose'][:, 3], aug_pose[:, 3])
			if ang_dist < min_ang_dist:
				min_ang_dist = ang_dist
				min_idx = ii
		img = cv2.imread(image_paths[min_idx])
		warped_img = warp_image(img, poses[min_idx], aug_pose, pixtocams)
		print(np.max(img))
		print(np.max(warped_img))
		cv2.imwrite('warped.png', warped_img)
		cv2.imwrite('origin.png', img)
	aug_poses = np.stack(aug_poses, axis=0)
	np.save(os.path.join(args.in_dir, "poses_bounds_aug.npy"), aug_poses)

	# q = rotation_quaternion(config.aug_roll, config.aug_pitch, config.aug_yaw)

	# warped_image = warp_image(image, np.linalg.inv(q.rotation_matrix), np.linalg.inv(test_dataset.pixtocams))

