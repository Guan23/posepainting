#!usr/bin/env python
# _*_ encoding: utf-8 _*_
# Author: Guan
# Create Time: 2021/12/15 上午10:43

import os
import numpy as np
import cv2

import myconfig as mcfg
from inference import preprocess, image_flow
import calibration_kitti
from object3d_kitti import get_objects_from_label

__all__ = ["get_lidar", "get_images_infos", "get_label", "get_calib_fromfile", "get_trans_matrix"]

args, box_model, pose_model, pose_transform, csv_output_rows = preprocess()  # 第一次加载pth权重


# 获取lidar原始数据，kitti的点云文件是training/velodyne/*.bin文件
def get_lidar(idx):
    lidar_file = mcfg.ROOT_SPLIT_PATH + 'velodyne/' + ('%s.bin' % idx)
    # np.fromfile：从文本或二进制文件中的数据构造一个数组
    return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)  # kitti的点云有四维，xyzi，第四维是intensity


# 读取左右两侧的图像并且获取左右图片得到的关节点信息
# TODO：暂时只用左侧相机
def get_images_infos(idx):
    image_path = os.path.join(mcfg.ROOT_SPLIT_PATH, "image_2", ("%s.png" % idx))
    pred_infos = image_flow(args, idx, box_model, pose_model, pose_transform, csv_output_rows, image_path)
    if pred_infos is None:
        return None, None, None
    image_gray = cv2.imread(image_path, 0)
    image_rgb = cv2.imread(image_path)
    image_mask = np.zeros_like(image_gray)

    # 返回图像、人体预测框、关节点坐标
    return image_rgb, image_mask, pred_infos


# 获得kitti已知的矫正信息calibration，其实主要用到的是相机内参和相机与激光雷达的外参（就是相对位置），返回的是calibration文件路径
def get_calib(idx):
    calib_file = mcfg.ROOT_SPLIT_PATH + 'calib/' + ('%s.txt' % idx)
    return calibration_kitti.Calibration(calib_file)


# 获取label信息（其实只用到了类别、遮挡、xyz，其他没用到）
def get_label(idx):
    label_file = mcfg.ROOT_SPLIT_PATH + 'label/' + ('%s.txt' % idx)
    # assert label_file.exists()
    return get_objects_from_label(label_file)


# 从上面得到的calib文件路径中，取得具体用到的calib参数，返回填充好的转换矩阵
def get_calib_fromfile(idx):
    # dict_keys(['P2', 'P3', 'R0', 'Tr_velo2cam', 'R0_rect'])
    calib_file = mcfg.ROOT_SPLIT_PATH + 'calib/' + ('%s.txt' % idx)
    calib = calibration_kitti.get_calib_from_file(calib_file)
    calib['P2'] = np.concatenate([calib['P2'], np.array([[0., 0., 0., 1.]])], axis=0)
    calib['P3'] = np.concatenate([calib['P3'], np.array([[0., 0., 0., 1.]])], axis=0)
    calib['R0_rect'] = np.zeros([4, 4], dtype=calib['R0'].dtype)
    calib['R0_rect'][3, 3] = 1.
    calib['R0_rect'][:3, :3] = calib['R0']
    calib['Tr_velo2cam'] = np.concatenate([calib['Tr_velo2cam'], np.array([[0., 0., 0., 1.]])], axis=0)
    return calib  # 返回字典，其中各个转换矩阵都填充成了4*4的形状


# 根据读取的calib文件信息，计算转换矩阵
def get_trans_matrix(idx):
    calib = get_calib_fromfile(idx)
    Trans_Matrix_left = calib["P2"].dot(calib["R0_rect"]).dot(calib["Tr_velo2cam"])
    # Trans_Matrix_right = calib["P3"].dot(calib["R0_rect"].dot(calib["Tr_velo2cam"]))
    return Trans_Matrix_left


if __name__ == "__main__":
    print("\n--------------- start ---------------\n")

    import copy
    import glob

    def points_3d_to_2d(points_3d, Trans_Matrix, image):
        lidar_points_3d = copy.deepcopy(points_3d)
        points_cam = (Trans_Matrix.dot(lidar_points_3d.transpose())).transpose()  # 得到相机坐标系下的(u，v，w)
        points_2d = points_cam / (points_cam[:, 2].reshape(-1, 1))  # (int(u/w)，int(v/w)才是最终的图像像素
        # 查看x和y坐标都落到图像内的点，过滤掉投影在语义图以外的点
        true_where_x_on_img = (0 < points_2d[:, 0]) & (points_2d[:, 0] < image.shape[1])
        true_where_y_on_img = (0 < points_2d[:, 1]) & (points_2d[:, 1] < image.shape[0])
        true_where_point_on_img = true_where_x_on_img & true_where_y_on_img
        # 各个列表都要更新，防止下标不对应
        points_2d = points_2d[true_where_point_on_img][:, 0:2]
        return len(lidar_points_3d), len(points_2d)

    data_folder = "G:/Kitti/ped"
    os.path.join(data_folder, "image_2")
    os.path.join(data_folder, "calib")
    os.path.join(data_folder, "velodyne")

    print("\n---------------- end ----------------\n")
