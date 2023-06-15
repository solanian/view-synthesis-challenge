import numpy as np
import matplotlib.pyplot as plt
import os
import json

import os, sys
import argparse

import matplotlib.animation as animation

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
    # Load camera poses
    poses, file_paths = load_poses(transforms_train_path)

    blender2opencv = np.array([[1,  0,  0,  0], 
                               [0, -1,  0,  0],
                               [0,  0, -1,  0],
                               [0,  0,  0,  1]])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # initializing a line variable
    line, = ax.plot(   [], 
                       [])

    for idx in range(0, len(file_paths)):
        pose_in_blender = poses[idx]
        pose_in_opencv = np.matmul(pose_in_blender, blender2opencv)
        
        cam_centre_x, cam_centre_y, cam_centre_z, tx, ty, tz \
            = get_points_to_visualise_camera_axis (pose_in_opencv, x_=.1, y_=.1, z_=.1)

        colors=['blue', 'green', 'red', 'red'] 
        for i_ in reversed(range(4)):
            if(i_ < 3): linestyle = 'solid'; alpha=1
            else:  linestyle = 'solid'; alpha=0.1

            ax.plot(   [cam_centre_x[i_], tx[i_]], 
                       [cam_centre_y[i_], ty[i_]],
                    zs=[cam_centre_z[i_], tz[i_]], 
                    color=colors[i_],
                    linestyle=linestyle,
                    alpha=alpha )
    
    # plt.show()

    if(args.save_jpg == True):
        # Rotate the axes and update
        elev, azim, roll = 45, -45, 45 ## sample viewpoint
        # Update the axis view and title
        ax.view_init(elev, azim, roll)
        plt.title('Elevation: %ddegree, Azimuth: %ddegree, Roll: %ddegree' % (elev, azim, roll))

        plt.draw()
        plt.savefig(args.out_path[:-4]+'.jpg')

        print('Saved file:', args.out_path[:-4]+'.jpg', '\n')

    else:
        anim = animation.FuncAnimation(fig, animate, 40, 
                            interval=1, blit=True)
        
        # saving to mp4 using ffmpeg writer
        writervideo = animation.FFMpegWriter(fps=10)
        anim.save(args.out_path, writer=writervideo)
        plt.close()

        print('Saved file:', args.out_path, '\n')