#!usr/bin/env python
# _*_ encoding: utf-8 _*_
# Author: Guan
# Create Time: 2021/11/16 下午15:28

# 系统库
import numpy as np
import cv2
import copy
import time
import csv
import logging

# 第三方库
import mayavi.mlab as mlab
import visualize_utils as V
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy import stats

# 自定义的函数
import inference
from load_data import *
import dilation
import myconfig as mcfg

__all__ = ["main"]


# -------------------------------------------------------------------

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
    
    # 暂时只用到了points_2d和w_axis，即投影到图像上的点和相机坐标系下的深度值
    return lidar_points_3d, points_cam, points_2d, w_axis, intensity,


def points_2d_to_3d(points_2d, Trans_Matrix, w_axis):
    '''
    @function: 2d点升维至3d
    @param points_2d: 2d点云
    @param Trans_Matrix: 转换矩阵（方阵）
    @param w_axis: w坐标轴的数据（降维时保留的）
    @return points_3d: 升维后的3d点云
    '''
    # 先填两列，分别是w坐标系和intensity信息，w的初始值为1
    points_d = np.insert(points_2d, points_2d.shape[1], values=1, axis=1)
    points_d_i = np.insert(points_d, points_d.shape[1], values=0, axis=1)
    points_3d = points_d_i * (w_axis.reshape(-1, 1))  # 注意这里乘w轴而非原始的z轴
    Trans_Matrix_inv = np.linalg.inv(Trans_Matrix)  # 转换矩阵求逆矩阵
    points_3d = (Trans_Matrix_inv.dot(points_3d.transpose())).transpose()  # 逆变换
    points_3d[:, -1] = mcfg.VIS_REFL  # 赋予反射强度信息，设置定值方便显示
    return points_3d


def draw_points_to_image(image, points_2d, pred_boxes, pose_preds):
    '''
    @function:在图像上画出对应的2D点，以及人体检测框和关节点
    @param image: 图像
    @param points_2d:转换后的2D点(float)
    @param pred_boxes_l: 人体检测框(float)
    @param pose_preds_l: 人体关节点(float)
    @return image: 绘制后的图像
    '''
    # 因为要对应像素位置，故要取整，这里选择向下取整，就不用考虑边界溢出情况，对吧
    points_2d = np.floor(points_2d).astype(int)
    # 舍弃后两维（原来的z和r，向下取整后变为了1和0），因为已经没用了，现在矩阵变成了 N*2 的形状
    points_2d = points_2d[:, :2]
    # 画人体框和框的序号
    for i, box in enumerate(pred_boxes):
        x1 = int(box[0][0])
        y1 = int(box[0][1])
        x2 = int(box[1][0])
        y2 = int(box[1][1])
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 69, 0), 2)
        cv2.putText(image, str(i + 1), (x1 + 5, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (102, 102, 255), 2)
    # 画激光点到图像上
    # colormap = (127, 255, 0)
    for point in points_2d:
        x_coord, y_coord = int(point[0]), int(point[1])
        cv2.circle(image, (x_coord, y_coord), 0, (127, 255, 0), -1)
    # 画人体关节点到图像上
    # for pose in pose_preds:
    #     for coord in pose:
    #         x, y = int(coord[0]), int(coord[1])
    #         cv2.circle(image, (x, y), 3, (238, 130, 238), 2)
    if mcfg.VISIBLE:
        inference.show_image(image)
    return image


# 给定2d框的中心center以及宽和高，判定point是否在矩形内部
def in_box(center, width, height, point):
    left_top_x = center[0] - width / 2.0
    left_top_y = center[1] - height / 2.0
    right_bottom_x = center[0] + width / 2.0
    right_bottom_y = center[1] + height / 2.0
    return point[0] > left_top_x and point[1] > left_top_y and point[0] < right_bottom_x and point[1] < right_bottom_y


def create_one_man_joint_3d_knn(image_mask, joint_points_2d, points_2d, Trans_Matrix, w_axis, scaler):
    joint_nums = len(joint_points_2d)
    candidate_human_points_2d = []
    candidate_depth = []
    # depth_array = np.zeros(shape=(joint_nums,))  # 17个关节点的深度（w坐标值）
    # 对所有的关节点使用KNN算法，挑选最临近的K个点云，选择其深度值赋予该关节点
    points_2d_int = np.floor(points_2d).astype(int)  # 向下取整，防止越界
    for i, lidar_point_2d in enumerate(points_2d_int):
        c_val = image_mask[lidar_point_2d[1]][lidar_point_2d[0]]  # mask上对应的channel值，背景为0，其余1～10为10个身体部位
        if c_val > 0:  # 落到人体掩膜上的点
            candidate_human_points_2d.append(lidar_point_2d)  # 各个身体部分对应的激光点分别存入各自的列表中
            candidate_depth.append(w_axis[i])  # 保持下标统一
    candidate_human_points_2d_np = np.array(candidate_human_points_2d)
    candidate_depth_np = np.array(candidate_depth)

    if candidate_human_points_2d_np.shape[0] == 0:
        return None
    samples = np.concatenate((joint_points_2d, candidate_human_points_2d_np), axis=0)  # 把17个关机点坐标添加到数组前面

    scaler.fit(samples)  # 归一化
    sampls_norm = scaler.transform(samples)

    nbrs = NearestNeighbors(n_neighbors=mcfg.KNN,
                            radius=0.5,
                            algorithm="auto",
                            leaf_size=30,
                            metric="minkowski",
                            p=2).fit(sampls_norm)
    # 返回距离每个点k个最近的点和距离指数，indices可以理解为表示点的下标，distances为距离
    distances, indices = nbrs.kneighbors(sampls_norm)
    ori_indices = indices[:joint_nums, 1:] - joint_nums  # 原来所有点的下标
    max_ori_indice = ori_indices.max()
    # knn可能会将关节点也视为邻居中的某个点，所以要把邻居中的关节点给剔除，换成点云
    # 如果最大下标依然为-17～-1,说明所有邻居都是关节点，需要删除此样本，否则就把邻居中的关节点换成最大下标那个点云
    if max_ori_indice < 0:
        return None
    else:
        ori_indices[ori_indices < 0] = max_ori_indice
        depth_array = np.zeros(shape=(joint_nums,))
        for i in range(mcfg.KNN - 1):
            depth_array += candidate_depth_np[ori_indices[:, i]]
        depth_array /= (mcfg.KNN - 1)
    joint_points_3d = points_2d_to_3d(joint_points_2d, Trans_Matrix, depth_array)
    return joint_points_3d


# 生成一个人的3D关节点，位置（xyz坐标），距离（xy的二范数），朝向（orientation）
# TODO:算法核心函数，关节点赋予身体区域的深度时，四肢是否可以更精确些？目前是用大腿代替膝盖，大臂代替肘这样子
def create_one_man_joint_3d(image_mask, joint_points_2d, points_2d, Trans_Matrix, w_axis, box, scaler):
    # 获取行人框中心位置
    x1 = box[0][0]
    y1 = box[0][1]
    x2 = box[1][0]
    y2 = box[1][1]
    box_size = [x2 - x1, y2 - y1]  # width, height
    center_point = np.array([[(x1 + x2) / 2.0, (y1 + box_size[1] * mcfg.BODY_Y)]])

    if mcfg.DETAILS:
        print("\n---center_point_xy_coord---:{}".format(center_point))

    joint_nums = len(joint_points_2d)
    candidate_center_depth = []  # 候选中心点深度值
    candidate_pose_depth = [[], [], [], [], [], [], [], [], [], []]  # 候选各个身体区域的深度值（列表）
    candidate_depth = []  # 候选最终各个身体区域的深度值
    points_2d_int = np.floor(points_2d).astype(int)  # 向下取整，防止越界
    # 对所有的框内的激光雷达点，选出各个身体部分上的候选点，以及距离在中心框之内的点
    for i, lidar_point_2d in enumerate(points_2d_int):
        c_val = image_mask[lidar_point_2d[1]][lidar_point_2d[0]]  # mask上对应的channel值，背景为0，其余1～10为10个身体部位
        if c_val > 0:  # 落到人体掩膜上的点
            candidate_pose_depth[c_val - 1].append(w_axis[i])  # 各个身体部分对应的激光点分别存入各自的列表中
        # 位于中心框内的点
        if in_box(center_point[0], mcfg.WIDTH_RANGE * box_size[0], mcfg.HEIGHT_RANGE * box_size[1], lidar_point_2d):
            candidate_center_depth.append(w_axis[i])
    if len(candidate_center_depth) == 0:
        print("the length of candidate_center_depth is 0, return None")
        return None
    # 转成numpy格式，后面计算更快
    if not isinstance(candidate_center_depth, np.ndarray):
        candidate_center_depth = np.array(candidate_center_depth)
    if not isinstance(candidate_pose_depth, np.ndarray):
        candidate_pose_depth = np.array(candidate_pose_depth, dtype=object)

    # 中心点的深度由聚类获得
    if mcfg.CLUSTER == "kmeans" and len(candidate_center_depth) >= mcfg.KMEANS:
        candidate_center_depth_X = np.expand_dims(candidate_center_depth, 1)
        scaler.fit(candidate_center_depth_X)
        candidate_center_depth_res = scaler.transform(candidate_center_depth_X)
        yp = KMeans(n_clusters=2).fit_predict(candidate_center_depth_res)
        center_depth = np.mean(candidate_center_depth[yp == stats.mode(yp)[0][0]])  # 众数的平均数
    elif mcfg.CLUSTER == "dbscan":
        candidate_center_depth_X = np.expand_dims(candidate_center_depth, 1)
        scaler.fit(candidate_center_depth_X)
        candidate_center_depth_res = scaler.transform(candidate_center_depth_X)
        min_samples = max(1, len(candidate_center_depth) // 16)
        yp = DBSCAN(eps=0.5, min_samples=min_samples).fit_predict(candidate_center_depth_res)
        center_depth = np.mean(candidate_center_depth[yp == stats.mode(yp[yp != -1])[0][0]])  # 众数的平均数
    else:
        center_depth = np.median(candidate_center_depth)

    if mcfg.DETAILS:
        print("candidate_center_depth:{}".format(candidate_center_depth))
        print("--------median_center_depth:{} --------".format(center_depth))
        print("--------candidate_center_depth_len:{} --------\n".format(len(candidate_center_depth)))

    # 对10个身体部位深度的列表进行筛选，同样，如果众数与中心点深度值的差别在一定值以内就选众数，否则就选中心点深度值
    for j, pose_depth in enumerate(candidate_pose_depth):
        if len(pose_depth) <= 2:
            candidate_depth.append(center_depth)
            continue
        depth_numpy = np.array(pose_depth)
        depth_numpy_X = np.expand_dims(depth_numpy, 1)
        scaler.fit(depth_numpy_X)
        depth_numpy_res = scaler.transform(depth_numpy_X)
        if mcfg.CLUSTER == "kmeans" and len(pose_depth) >= mcfg.KMEANS:
            y_pred = KMeans(n_clusters=mcfg.KMEANS).fit_predict(depth_numpy_res)
            means = []
            for i in range(mcfg.KMEANS):
                mean = np.mean(depth_numpy[y_pred == i])
                if abs(mean - center_depth) <= mcfg.DEPTH_BIAS:
                    means.append(mean)
            mean_val = np.mean(np.array(means)) if len(means) > 0 else center_depth  # 平均数
            mode_val = np.mean(depth_numpy[y_pred == stats.mode(y_pred)[0][0]])  # 众数的平均数
            mode_val = mode_val if abs(mode_val - center_depth) <= mcfg.DEPTH_BIAS else center_depth

            candidate_depth.append(mean_val)
        elif mcfg.CLUSTER == "dbscan":
            min_samples = max(1, len(depth_numpy) // 8)
            y_pred = DBSCAN(eps=0.1, min_samples=min_samples).fit_predict(depth_numpy_res)
            # 只保留与中心点的深度差值在一定范围内，并且不是噪点的点
            is_in_range = (depth_numpy - center_depth <= mcfg.DEPTH_BIAS) & (y_pred != -1)
            # y_pred = y_pred[is_in_range]
            depth_numpy_sample = depth_numpy[is_in_range]
            if len(depth_numpy_sample) > 0:
                y_pred = y_pred[is_in_range]
                mean_val = np.mean(np.array(depth_numpy_sample))
                mode_val = np.mean(depth_numpy_sample[y_pred == stats.mode(y_pred)[0][0]])
            else:
                mean_val = center_depth
            candidate_depth.append(mean_val)
        else:
            candidate_depth.append(center_depth)
            continue

        if mcfg.DETAILS:
            print("the {} body area's list length is {}".format(j, len(pose_depth)))

    # 创造深度数组和反射强度数组，用于计算
    depth_array = np.ones(joint_nums)  # (17, )
    depth_array[0:5] = candidate_depth[0]  # 鼻子、双眼、双耳---头部
    depth_array[5:7] = candidate_depth[1]  # 双肩---躯干
    depth_array[11:13] = candidate_depth[1]  # 双髋---躯干
    depth_array[7] = (candidate_depth[2] + candidate_depth[3]) / 2  # 左肘---左大臂
    depth_array[9] = candidate_depth[3]  # 左腕---左小臂
    depth_array[8] = (candidate_depth[4] + candidate_depth[5]) / 2  # 右肘---右大臂
    depth_array[10] = candidate_depth[5]  # 右腕---右小臂
    depth_array[13] = (candidate_depth[6] + candidate_depth[7]) / 2  # 左膝---左大腿
    depth_array[15] = candidate_depth[7]  # 左踝---左小腿
    depth_array[14] = (candidate_depth[8] + candidate_depth[9]) / 2  # 右膝---右大腿
    depth_array[16] = candidate_depth[9]  # 右踝---右小腿

    # refl_array = np.ones(joint_nums)  # 反射率
    joint_points_3d = points_2d_to_3d(joint_points_2d, Trans_Matrix, depth_array)
    return joint_points_3d


# 生成一张图中所有人的3D关节点
def create_one_image_joint_3d(joint_points_2d, points_3d, Trans_Matrix, image_mask, bbox, scaler):
    lidar_points_3d, points_cam, points_2d, w_axis, intensity = points_3d_to_2d(points_3d, Trans_Matrix, image_mask)
    # points_2d = points_2d[:, :2]  # 舍弃掉后两维
    joint_points_3d_list = []
    # 只保留人体检测框内的信息
    for i, box in enumerate(bbox):
        x1 = box[0][0]
        y1 = box[0][1]
        x2 = box[1][0]
        y2 = box[1][1]
        # box_size = ((x2 - x1), (y2 - y1))  # width, height
        true_where_x_in_box = (x1 < points_2d[:, 0]) & (points_2d[:, 0] < x2)
        true_where_y_in_box = (y1 < points_2d[:, 1]) & (points_2d[:, 1] < y2)
        true_where_point_in_box = true_where_x_in_box & true_where_y_in_box
        # 注意，这里筛选的时候，要把所有信息数组都进行筛选，不然下标就不对应流
        # 并且注意不要修改原array，而是另存一个array做实参
        points_2d_in_box = points_2d[true_where_point_in_box]
        points_3d_in_box = lidar_points_3d[true_where_point_in_box]
        w_axis_in_box = w_axis[true_where_point_in_box]
        intensity_in_box = intensity[true_where_point_in_box]
        points_cam_in_box = points_cam[true_where_point_in_box]

        # 选择合适的深度赋值算法为一个行人的所有关节点赋值
        if mcfg.ASSIGN_ALGORITHM == "mask":
            joint_points_3d = create_one_man_joint_3d(image_mask,
                                                      joint_points_2d[i],
                                                      points_2d_in_box,
                                                      Trans_Matrix,
                                                      w_axis_in_box,
                                                      box,
                                                      scaler)
        elif mcfg.ASSIGN_ALGORITHM == "knn":
            joint_points_3d = create_one_man_joint_3d_knn(image_mask,
                                                          joint_points_2d[i],
                                                          points_2d_in_box,
                                                          Trans_Matrix,
                                                          w_axis_in_box,
                                                          scaler)
        else:
            logging.error("------ 请在mask和knn中选择一种算法！！ ------")
            raise KeyError("ASSIGN_ALGORITHM must be one of mask or knn")
        if joint_points_3d is not None:
            joint_points_3d_list.append(joint_points_3d)
    print("\n")
    return joint_points_3d_list


# 生成关键关节点（筛选了6个主要的关节点），已弃用
def create_critical_pose(pose_preds):
    critical_pose = []
    critical_pose.append(pose_preds[:, 5, :])  # 左肩
    critical_pose.append(pose_preds[:, 6, :])  # 右肩
    critical_pose.append(pose_preds[:, 11, :])  # 左胯
    critical_pose.append(pose_preds[:, 12, :])  # 右胯
    critical_pose.append(pose_preds[:, 13, :])  # 左膝
    critical_pose.append(pose_preds[:, 14, :])  # 右膝
    if not isinstance(critical_pose, np.ndarray):
        critical_pose = np.array(critical_pose)  # list to numpy
    critical_pose = np.transpose(critical_pose, (1, 0, 2))  # 更换维度
    return critical_pose


# 输出可视化的三维人体姿态点，去掉反射率一维，双肩和双髋中间各取一个中点作为17和18号点
# 把人整体xyz移动到0以上（减最小值）
def create_joint_3d_vis(joint_points_3d_list):
    joint_points_3d_vis = joint_points_3d_list[:, :, 0:-1]
    up_mid = (joint_points_3d_vis[:, 5, :] + joint_points_3d_vis[:, 6, :]) / 2.0
    down_mid = (joint_points_3d_vis[:, 11, :] + joint_points_3d_vis[:, 12, :]) / 2.0
    joint_points_3d_vis = np.insert(joint_points_3d_vis, joint_points_3d_vis.shape[1], up_mid, axis=1)
    joint_points_3d_vis = np.insert(joint_points_3d_vis, joint_points_3d_vis.shape[1], down_mid, axis=1)
    # 每一维度减去最小值或平均值
    min_joint = np.mean(joint_points_3d_vis, axis=1, keepdims=True)
    # if not mcfg.SINGLE:
    min_joint[:, :, 0:-1] = 0.0
    joint_points_3d_vis = joint_points_3d_vis - min_joint
    return joint_points_3d_vis


# 把所有人的关节点信息添加到原始点云上，共同显示
def concat_visual(joint_points_3d_list, lidar_points):
    lidar_points_vis = lidar_points.copy()
    lidar_points_vis[:, -1] = 1  # 把原始点云的反射率全部置1,方便显示
    for joint_points in joint_points_3d_list:
        lidar_points_vis = np.concatenate((lidar_points_vis, joint_points), axis=0)
    if mcfg.VISIBLE:
        if mcfg.MATVIS:
            joint_points_3d_vis = create_joint_3d_vis(joint_points_3d_list)
            V.draw_pose(points=joint_points_3d_vis)
        else:
            V.draw_scenes(points=lidar_points_vis, point_size=0.05)
            mlab.show(stop=True)


# 绘制一张图中所有人的轮廓，输出的image_mask用于计算，image_rgb用于可视化
def draw_one_image_outline(pred_boxes, pose_preds, image_mask, image_rgb):
    for i, pose_pred in enumerate(pose_preds):
        x1 = pred_boxes[i][0][0]
        y1 = pred_boxes[i][0][1]
        x2 = pred_boxes[i][1][0]
        y2 = pred_boxes[i][1][1]
        box_size = ((x2 - x1), (y2 - y1))  # width, height
        human_size = dilation.get_human_proportion(pose_pred, box_size)
        if mcfg.RECORD_CSV:
            # 写入csv文件
            test_dir = "/home/gxk/HRnet/human_size/test.csv"
            rows = [
                [box_size[0], box_size[1], pose_pred[0][0], pose_pred[0][1],
                 pose_pred[3][0], pose_pred[3][1], pose_pred[4][0], pose_pred[4][1],
                 human_size[0][0], human_size[0][1], human_size[1]],
            ]
            with open(test_dir, 'a+') as f:
                f_csv = csv.writer(f)
                f_csv.writerows(rows)
        if mcfg.DETAILS:
            print("\n-----------------person number-----------------:{}\n".format(i + 1))
            # print("pose:{}".format(pose_preds[i]))
            print("box_size:{}".format(box_size))
            print("y value:{},{},{},{}".format(human_size[0], human_size[1], human_size[6], human_size[7]))
            print("-------human_size-------:{}".format(human_size))
        image_mask = dilation.draw_outline(image_mask, pose_pred, dilation.CHANNEL_VAL_, human_size)

        # 只是为了可视化，实际中测速度时可以注释掉这里
        image_rgb = dilation.draw_outline(image_rgb, pose_pred, dilation.COLOR_, human_size)
    return image_mask, image_rgb


def main(idx):
    # 获取原始点云和转换矩阵信息
    t0 = time.time()
    raw_lidar = get_lidar(idx)
    # 只保留前向（0m, 80m）范围内的点
    true_where_lidar_in_range = (0 < raw_lidar[:, 0]) & (raw_lidar[:, 0] < 80)
    near_lidar = raw_lidar[true_where_lidar_in_range]
    Trans_Matrix = get_trans_matrix(idx)
    if mcfg.DETAILS:
        print("\n------------------------------\n")
        print("raw_lidar_length:{}".format(len(raw_lidar)))
        print("near_lidar_length:{}".format(len(near_lidar)))
        print("\n------------------------------\n")
    t1 = time.time()
    image_rgb, image_mask, pred_infos = get_images_infos(idx)

    if pred_infos is None:
        print("----------There are no humans in this picture!!!----------")
        return None, None
        # exit(1)

    pred_boxes = np.array(pred_infos[0])  # 人体检测框(N, 2, 2)
    pose_preds = np.array(pred_infos[1])  # 人体关节点(N, 17, 2)

    if mcfg.DETAILS:
        print('---2d_pred_boxes.shape:{}---'.format(pred_boxes.shape))
        print('---2d_pose_preds.shape:{}---\n'.format(pose_preds.shape))

    t2 = time.time()
    # 画人体轮廓
    image_mask, image_rgb = draw_one_image_outline(pred_boxes, pose_preds, image_mask, image_rgb)
    if mcfg.VISIBLE:
        inference.show_image(image_rgb)
    # 生成3d关节点
    # critical_pose = create_critical_pose(pose_preds)  # 关键关节点(N, 7, 2)
    lidar_points_3d, points_cam, points_2d, w_axis, intensity = points_3d_to_2d(near_lidar, Trans_Matrix, image_mask)
    t3 = time.time()
    scaler = MinMaxScaler(feature_range=(0, 1))
    joint_points_3d_list = create_one_image_joint_3d(pose_preds, near_lidar, Trans_Matrix, image_mask, pred_boxes,
                                                     scaler)
    if len(joint_points_3d_list) == 0:
        return None, None
    joint_points_3d_list = np.array(joint_points_3d_list)
    joint_points_3d_list = joint_points_3d_list[:] + mcfg.JOINT_3D_BIAS  # 添加上人体的厚度信息
    t4 = time.time()
    if mcfg.DETAILS:
        print("-----joint_points_3d_list.shape:{}-----\n".format(joint_points_3d_list.shape))
        print("-----joint_points_3d_list:\n{}-----\n".format(joint_points_3d_list))
        print("-----读取数据耗时: {:.5f} sec-----".format(t1 - t0))
        print("-----目标检测和二维人体姿态网络推理耗时: {:.5f} sec-----".format(t2 - t1))
        print("-----点云正投影到图像耗时: {:.5f} sec-----".format(t3 - t2))
        print("-----深度筛选赋值+逆投影耗时: {:.5f} sec-----".format(t4 - t3))
        print("-----单独的升维模块耗时: {:.5f} sec-----".format(t4 - t2 + t1 - t0))
        print("-----整体耗时: {:.5f} sec-----".format(t4 - t0))
    concat_visual(joint_points_3d_list, lidar_points_3d)  # 点云可视化
    draw_points_to_image(image_rgb, points_2d, pred_boxes, pose_preds)  # 多传感器信息融合可视化
    return pred_boxes, joint_points_3d_list


if __name__ == '__main__':
    print('\n-----------------------start-----------------------\n')

    # TODO：2021_12_14_11_45
    # 4、点归属判断那边，超过20个或者30个就break行了，再多也是取一个值，没必要
    # 6、对数据帧按kitti分难度测评
    # 7、评估模块多加一个CDE，即center_depth_error，只统计深度维的误差
    # 8、评估模块再加一个ALA，即距离误差分段的比例，三个值分别是ALA>0.5,1,2m
    # 9、生成人体掩膜的代码优化，固定参数改成与2d box相关的变量
    # 10、生成3d loc的代码改改深度值的偏移值，改成变量（变量的变化不大，因为这是3D下真实的深度距离）
    # 11、漏检太多了，查查是怎么回事
    args, box_model, pose_model, pose_transform, csv_output_rows = inference.preprocess()  # 第一次加载pth权重

    # 单张测试
    idx = "000000"
    pred_boxes, joint_points_3d_list = main(idx)

    print('\n------------------------end------------------------\n')
