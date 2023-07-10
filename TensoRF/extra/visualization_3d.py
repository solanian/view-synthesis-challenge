# -*- coding: utf-8 -*-
import csv
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    # for csv_file_path in ["one_cam_points_cloud.csv","multi_cam_points_cloud.csv"]:
    for csv_file_path in ["one_cam_points_cloud.csv"]:
        f = open(csv_file_path, 'r')
        csv_list = np.array(list(csv.reader(f)))
        f.close()
        point_clouds = []
        x, y, z, L = [],[],[],[]
        rgb = []
        cam_loc_list = []
        view_edge_list = []
        for i in range(csv_list.shape[0]):
            if csv_list[i][7] == "cam_loc":
                cam_loc_list.append([float(csv_list[i][0]), float(csv_list[i][1]), float(csv_list[i][2])])
            elif csv_list[i][7] == "view_dir":
                view_edge_list.append([float(csv_list[i][0]), float(csv_list[i][1]), float(csv_list[i][2])])

            x.append(float(csv_list[i][0]))
            y.append(float(csv_list[i][1]))
            z.append(float(csv_list[i][2]))
            L.append(float(csv_list[i][3]))

            rgb_val = [float(csv_list[i][4]), float(csv_list[i][5]), float(csv_list[i][6])]
            alpha_check_list = np.array(rgb_val) * 255
            alpha_check_list = [int(alpha_check_list[0]), int(alpha_check_list[1]), int(alpha_check_list[2])]
            if all(alpha_check_list) != 0:
                rgb_val.append(1.0)
            elif alpha_check_list == [255, 255, 0]:
                rgb_val.append(0.02)
            else:
                rgb_val.append(0.2)
            rgb.append(rgb_val)

        fig = plt.figure(figsize=(20, 20))
        ax = plt.axes(projection="3d")

        view_edge_np = np.array(view_edge_list).reshape(-1,4,2,3)
        if csv_file_path == "multi_cam_points_cloud.csv":
            cam_idx_list = list(range(len(cam_loc_list)))
        else:
            cam_idx_list = [0]

        d = [0,1]
        for cam_idx in cam_idx_list:
            plot_x, plot_y, plot_z, plot_rgb = [], [], [], []
            ven = view_edge_np[cam_idx]
            xyz = [ven[0][d[0]], ven[1][d[0]], ven[3][d[0]], ven[2][d[0]],
                   ven[2][d[1]], ven[3][d[1]], ven[1][d[1]], ven[0][d[1]],
                   ven[0][d[0]], ven[2][d[0]], ven[3][d[0]], ven[3][d[1]],
                   ven[2][d[1]], ven[0][d[1]], ven[1][d[1]], ven[1][d[0]]]
            for vertex in xyz:
                x.append(vertex[0])
                y.append(vertex[1])
                z.append(vertex[2])
                L.append(10.0)
                rgb.append([0, 0, 1.0, 1.0])
                plot_x.append(vertex[0])
                plot_y.append(vertex[1])
                plot_z.append(vertex[2])
            ax.plot(plot_x, plot_y, plot_z, c='r')

        ax.scatter3D(np.array(x, dtype=float),
                     np.array(y, dtype=float),
                     np.array(z, dtype=float),
                     c=np.array(rgb),
                     s=np.array(L))

        plt.show()
