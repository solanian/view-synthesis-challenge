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
        prop = []
        for i in range(csv_list.shape[0]):
            x.append(float(csv_list[i][0]))
            y.append(float(csv_list[i][1]))
            z.append(float(csv_list[i][2]))
            L.append(float(csv_list[i][3]))

            rgb_val = [float(csv_list[i][4]),float(csv_list[i][5]),float(csv_list[i][6])]
            # print(rgb_val)
            alpha_check_list = np.array(rgb_val) * 255
            # print(alpha_check_list)
            alpha_check_list = [int(alpha_check_list[0]), int(alpha_check_list[1]), int(alpha_check_list[2])]
            if all(alpha_check_list) != 0:
                rgb_val.append(1.0)
            elif alpha_check_list == [255,255,0]:
                rgb_val.append(0.02)
            else:
                rgb_val.append(0.2)
            rgb.append(rgb_val)
            prop.append(csv_list[i][7])

        fig = plt.figure(figsize=(10,7))
        ax = plt.axes(projection ="3d")
        ax.scatter3D(np.array(x,dtype=float),
                     np.array(y,dtype=float),
                     np.array(z,dtype=float),
                     c=np.array(rgb))
        plt.show()
