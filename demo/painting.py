import numpy as np
import os
import copy
from load_data import *
import myconfig as mcfg
import cv2


def COLORMAP_JET(i):
    if i < 0 or i > 255:
        return [0, 0, 0]
    if i >= 0 and i <= 31:
        return [128 + 4 * i, 0, 0]
    elif i == 32:
        return [255, 0, 0]
    elif i >= 33 and i <= 95:
        return [255, 4 + 4 * i, 0]
    elif i == 96:
        return [254, 255, 2]
    elif i >= 97 and i <= 158:
        return [250 - 4 * i, 255, 6 + 4 * i]
    elif i == 159:
        return [1, 255, 254]
    elif i >= 160 and i <= 223:
        return [0, 252 - 4 * i, 255]
    else:
        return [0, 0, 252 - 4 * i]


def points_3d_to_2d(points_3d, Trans_Matrix, image):
    '''
    @function:3d点云转成2d图像点，并把图像范围外的点云删除
    @param points_3d: 3d点云
    @param Trans_Matrix: 转换矩阵（方阵）
    @param image: 映射的图像（获取尺寸）
    @return:
        lidar_points_3d: 点云坐标系下的点
        points_cam: 相机坐标系下的点
        points_2d: 图像坐标系下的点
        w_axis: 相机坐标系下的w轴，即深度
        intensity: 点云坐标系下的反射强度
    '''
    lidar_points_3d = copy.deepcopy(points_3d)
    intensity = copy.deepcopy(points_3d[:, -1])  # 把反射强度信息深拷贝一份
    # lidar_points_3d[:, -1] = 1  # 把intensity那一列改成1,方便做齐次运算
    points_cam = (Trans_Matrix.dot(lidar_points_3d.transpose())).transpose()  # 得到相机坐标系下的(u，v，w)
    w_axis = points_cam[:, 2]  # 注意这个w坐标轴的数据需要保存，升维需要用到
    points_2d = points_cam / (points_cam[:, 2].reshape(-1, 1))  # (int(u/w)，int(v/w)才是最终的图像像素
    # 查看x和y坐标都落到图像内的点，过滤掉投影在语义图以外的点
    true_where_x_on_img = (0 < points_2d[:, 0]) & (points_2d[:, 0] < image.shape[1])
    true_where_y_on_img = (0 < points_2d[:, 1]) & (points_2d[:, 1] < image.shape[0])
    true_where_point_on_img = true_where_x_on_img & true_where_y_on_img
    # 各个列表都要更新，防止下标不对应
    points_cam = points_cam[true_where_point_on_img][:, 0:3]
    points_2d = points_2d[true_where_point_on_img][:, 0:2]
    intensity = intensity[true_where_point_on_img]
    w_axis = w_axis[true_where_point_on_img]
    lidar_points_3d = lidar_points_3d[true_where_point_on_img]

    return lidar_points_3d, points_cam, points_2d, w_axis, intensity


if __name__ == "__main__":
    print("\n----------------- start -----------------\n")
    idx = "000288"

    calib = get_calib_fromfile(idx)

    Trans_Matrix = get_trans_matrix(idx)
    raw_lidar = get_lidar(idx)
    true_where_lidar_in_range = (0 < raw_lidar[:, 0]) & (raw_lidar[:, 0] < 80)
    points_3d = raw_lidar[true_where_lidar_in_range]

    image_path = os.path.join(mcfg.ROOT_SPLIT_PATH, "image_2", ("%s.png" % idx))
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    points_proj = np.zeros((height, width, 1), dtype=np.uint8)

    lidar_points_3d, points_cam, points_2d, w_axis, intensity = points_3d_to_2d(points_3d, Trans_Matrix, image)

    maxdepth = np.max(lidar_points_3d[:, 0])
    mindepth = np.min(lidar_points_3d[:, 0])
    print(maxdepth)
    print(mindepth)

    for i, point in enumerate(points_2d):
        x_coord, y_coord = int(point[0]), int(point[1])
        depth = (lidar_points_3d[i, 0] - mindepth) / (maxdepth - mindepth)
        cv2.circle(image, (x_coord, y_coord), 0, color=COLORMAP_JET(np.floor(depth * 255)), thickness=1)

    cv2.imshow("image", image)
    key = cv2.waitKey(0) & 0xff
    if key == ord("s") or key == ord("S"):
        cv2.imwrite("./{}_save.png".format(idx), image)
        print("image saved!")
    cv2.destroyAllWindows()

    print("\n------------------ end ------------------\n")
