#!usr/bin/env python
# _*_ encoding: utf-8 _*_
# Author: Guan
# Create Time: 2021/11/16 下午12:29

'''
最开始写的实验函数，自己假设了一些3D点和图像，并且自己设定了外参矩阵，进行反投影实验
'''

import numpy as np
import mayavi.mlab as mlab

# ---------------------------------------------------------------
JOINT_SEARCH_RADIUS = 1.4  # 关节点的搜索半径，激光点与关节点的距离小于此半径的认为是同一物体
CENTER_SEARCH_RADIUS = 1.3  # 关节点中心的搜索半径，激光点与中心的距离小于此半径的认为是同一物体


# ---------------------------------------------------------------


def joint_2d_to_3d(joint_points_2d, lidar_points_3d, Trans_Matrix, bbox):
    '''
    :function:计算得到3D人体关节点
    :param joint_points_2d: 2d人体关节点
    :param lidar_points_3d: 3d点云
    :param R: 转换矩阵
    :param bbox: 人体检测框（x1，y1，x2，y2）
    :return joint_points_3d:转换后的人体3D关节点或者None
    '''
    min_joint_dis = JOINT_SEARCH_RADIUS  # 激光点和关节点的最小欧式距离
    # 1范数/第一维维数得到所有关节点的中心点，TODO:可以只要躯干点，抛弃四肢和头部
    center_point = np.linalg.norm(joint_points_2d, ord=1, axis=0, keepdims=True) / joint_points_2d.shape[0]
    depth = -1  # 激光原始点云的深度信息
    lidar_points_2d = lidar_3d_to_2d(lidar_points_3d, Trans_Matrix, bbox)  # 3d点云转成2d点云

    # 对所有的激光雷达点，先看与关节点中心的距离，再看与各个关节点的距离，都要在半径范围内才认定此点为人体点
    for i, lidar_point in enumerate(lidar_points_2d):
        if np.linalg.norm(lidar_point - center_point) < CENTER_SEARCH_RADIUS:
            for j, joint_point in enumerate(joint_points_2d):
                if np.linalg.norm(joint_point - lidar_point) < min_joint_dis:
                    min_joint_dis = np.linalg.norm(joint_point - lidar_point)  # 激光点与关节点的最小距离
                    depth = lidar_points_3d[i][1]  # 对应的深度信息

    if min_joint_dis < JOINT_SEARCH_RADIUS:
        joint_points_3d = lidar_2d_to_3d(lidar_points_2d, Trans_Matrix, depth)
        return joint_points_3d
    else:  # 远处的点云不加入计算，其实就相当于自动舍弃了此人
        return None


def lidar_3d_to_2d(lidar_points_3d, Trans_Matrix, bbox):
    '''
    :function:把原始点云经过变换矩阵R，转换为图像坐标系下对应的点
    :param lidar_points_3d: 原始的3d点云
    :param R: 转换矩阵
    :param bbox: 人体检测的2d框
    :return lidar_points_2d:转换后的图像坐标系下的点
    '''
    # TODO:判断矩阵形状能否相乘
    lidar_points_2d = (Trans_Matrix.dot(lidar_points_3d.transpose())).transpose()  # 坐标变换矩阵
    # np.delete(array,obj,axis)，axis是第几维，obj是该维度的第几维数
    # TODO:删除的维数其实在后面，即保留前两维
    lidar_points_2d = np.delete(lidar_points_2d, 1, 1)  # 删除第二列（y坐标）
    # 删除检测框以外的点
    for i, p in enumerate(lidar_points_2d):
        if p[0] < bbox[0][0] or p[0] > bbox[1][0] or p[1] < bbox[0][1] or p[1] > bbox[1][1]:
            lidar_points_2d = np.delete(lidar_points_2d, i, 0)
    return lidar_points_2d


def lidar_2d_to_3d(lidar_points_2d, Trans_Matrix, depth):
    '''
    :function:2d点经过逆变换升维至3d（深度那一维生成的数字无意义，需要使用真实的深度信息进行覆盖）
    :param lidar_points_2d: 2d点坐标
    :param R: 转换矩阵
    :param depth: 补充的深度信息
    :return lidar_points_3d:补充了深度信息后的3d点
    '''
    # TODO:lidar_points_2d需要填充0,1，先升维
    Trans_Matrix_inv = np.linalg.inv(Trans_Matrix)  # 转换矩阵求逆
    lidar_points_2d = np.insert(lidar_points_2d, 1, values=1, axis=1)  # 升维
    lidar_points_3d = Trans_Matrix_inv.dot(lidar_points_2d.transpose())  # 逆向升到三维，但第二维度的数是没用的，需要覆盖
    lidar_points_3d = lidar_points_3d.transpose()
    lidar_points_3d[:, 1] = depth  # TODO：同理，增加的维数也要改一下
    return lidar_points_3d


if __name__ == '__main__':
    print('\n-----------------------start-----------------------\n')
    Trans_Matrix = np.eye(3)
    joint_points_2d = np.array(
        [[1, 4],
         [2, 3],
         [3, 2],
         [3, 3],
         [4, 3],
         [5, 3]]
    )
    lidar_points_3d = np.array(
        [[1, 20, 0],
         [1, 20, 1],
         [2, 20, 1],
         [2, 20, 2],
         [3, 10, 4]]
    )
    bbox = np.array(
        [[0, 0],
         [6, 6]]
    )

    # joint_points_2d = np.array(
    #     [[0, 0],
    #      [2, 1],
    #      [2, 2],
    #      [1, 0],
    #      [3, 0]]
    # )
    # lidar_points_3d = np.array(
    #     [[0, 13, 4],
    #      [1, 32, 2],
    #      [2, 15, 1],
    #      [2, 20, 3]]
    # )
    # bbox = np.array(
    #     [[0, 0],
    #      [4, 5]]
    # )

    joint_points_3d = joint_2d_to_3d(joint_points_2d, lidar_points_3d, Trans_Matrix, bbox)
    print(joint_points_3d)
    # mlab.points3d(joint_points_3d)
    # mlab.show()

print('\n------------------------end------------------------\n')
