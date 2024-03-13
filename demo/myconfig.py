#!usr/bin/env python
# _*_ encoding: utf-8 _*_
# Author: Guan
# Create Time: 2021/12/17 下午3:18

import numpy as np

# ======================= 使用中 =========================

ROOT_SPLIT_PATH = "../training_pedestrain/"  # kitti数据集的根目录
LOG_DIR = "../mylog"  # 存储log文件的文件夹
DETECT_MODEL = "mask_rcnn"  # 选择使用的2D行人目标检测网络，可选"faster_rcnn", "retina_net"或"mask_rcnn"
DETECT_SCORE_THRESH = 0.9  # 目标检测的分数阈值，score低于此数的框删掉
ASSIGN_ALGORITHM = "mask"  # 深度赋值算法，可选"knn"|"mask"
CLUSTER = "dbscan"  # 聚类算法选择，可选"dbscan"|"kmeans"
KNN = 2  # KNN最临近算法，选最近的几个点，需大于等于2，当赋值为2时，KNN退化到NLA算法
KMEANS = 2  # 如果使用KMeans聚类，选择的聚类种类数，默认6

# 因为对于挡住一半的人，可能真实框是人的全体，而预测框是半个人，此时IOU可能小于阈值，但我们也认为预测正确
# 判断两个框是否有包含关系，框的格式是x1y1,x2y2，即左上点和右下点
INVOLVE_WIDTH_THRES = -0.03  # 小框的x1 - 大框的x1 > 此值 * 大框的宽，x2同理
INVOLVE_HEIGHT_THRES = -0.04  # 小框的y1 - 大框的y1 > 此值 * 大框的高，y2同理
INVOLVE_AREA_THRES = 0.95  # 大框面积要与小框差不多，才认为是两个行人，如果两框面积相差太大，可能是误检

WIDTH_RANGE = 0.45  # 检测框的宽乘此比例，即选取中心点的范围宽，default: 0.4
HEIGHT_RANGE = 0.28  # 检测框的高乘此比例，即选取中心点的范围高，default: 0.3

DEPTH_BIAS = 0.5  # 各部位深度值与身体中心的深度值偏差在此范围内则保留，默认0.5米
BODY_DEPTH = 0.20  # # 激光雷达打的是表面，人体厚度默认是0.2米


# 人体掩膜相关参数，以标准八头身模型为例，头的长度即基线长度，身体各部位均以基线表示
BODY_WIDTH = 1.6  # 躯干宽度等于几个基线长度，默认1.6
BODY_HEIGHT = 3.0  # 躯干长度等于几个基线长度，默认3.0
BODY_Y = 0.2  # 人体框上边界 + 此比例 x 人体框的高度 = 躯干框中心y
HEAD_RATIO = 0.6  # 头部的半径等于几个基线长度，默认0.6
BIG_ARM_RATIO = 3.2  # 手臂长度与宽度的比值，默认3.2
SMALL_ARM_RATIO = 3.2  # 手臂长度与宽度的比值，默认3.2
BIG_LEG_RATIO = 3.5  # 腿部长度与宽度的比值，默认3.5
SMALL_LEG_RATIO = 3.5  # 腿部长度与宽度的比值，默认3.5

# 三种难度下行人目标检测的IOU阈值，预测框与真实框的IOU超过此阈值认为是同一样本,默认0.5
EASY_IOU_THRESH = 0.5
MODERATE_IOU_THRESH = 0.5
HARD_IOU_THRESH = 0.5

SAVE_LOG = True  # 批量评估时是否保存log文件
RECORD_CSV = False  # 是否将一些人体关节点坐标记录到csv文件中
VISIBLE = True  # 是否可视化
MATVIS = True  # 使用matplotlib可视化还是使用mayavi可视化，选择True使用matplotlib
DETAILS = False  # 是否打印中间数据
DILATION = False  # 消融实验

# KITTI数据集的一些常量
TOTAL_FRAME = 1779  # 所有包含pedestrian类的帧数
EASY_PED = 2325  # 简单样本
MODERATE_PED = 1258  # 中等样本
HARD_PED = 708  # 困难样本
TOTAL_PED = 4291  # 简单+中等+困难样本，注意还有少量样本难度属于未知，此部分样本不予考虑

# ALA评价指标的阈值，分成三档，[0, 0.5), [0, 1.0), [0, 2.0)，剩下的比例就是[2.0, +无穷)的
ALA_THRESH = [0.5, 1.0, 2.0]

# 人体17个关节点位置的厚度信息
# JOINT_3D_BIAS = np.array(
#     [[0.05, 0, 0, 0], [0.075, 0, 0, 0], [0.075, 0, 0, 0], [0.075, 0, 0, 0], [0.075, 0, 0, 0],
#      [0.08, 0, 0, 0], [0.08, 0, 0, 0],
#      [0.06, 0, 0, 0], [0.06, 0, 0, 0], [0.055, 0, 0, 0], [0.055, 0, 0, 0],
#      [0.08725, 0, 0, 0], [0.08725, 0, 0, 0],
#      [0.065, 0, 0, 0], [0.065, 0, 0, 0], [0.0575, 0, 0, 0], [0.0575, 0, 0, 0]]
# )

JOINT_3D_BIAS = np.array(
    [[0.0, 0, 0, 0], [0.0, 0, 0, 0], [0.0, 0, 0, 0], [0.0, 0, 0, 0], [0.0, 0, 0, 0],
     [0.0, 0, 0, 0], [0.0, 0, 0, 0],
     [0.0, 0, 0, 0], [0.0, 0, 0, 0], [0.0, 0, 0, 0], [0.0, 0, 0, 0],
     [0.0, 0, 0, 0], [0.0, 0, 0, 0],
     [0.0, 0, 0, 0], [0.0, 0, 0, 0], [0.0, 0, 0, 0], [0.0, 0, 0, 0]]
)

# 方便可视化设置的反射率，主要区分左右腿左右手，左深右浅
# VIS_REFL = np.array(
#     [0.1, 0.1, 0.1, 0.1, 0.1,
#      0.25, 0.25, 0.15, 0.65, 0.15, 0.65,
#      0.35, 0.35, 0.4, 0.75, 0.4, 0.75]
# )

VIS_REFL = np.array(
    [0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
)
# =======================================================


# ======================= 已弃用 =========================
# NECK_WIDTH = 2.5  # 脖子长度与宽度的比值（已弃用）
# EYE2NOSE = 3.12  # 整体脸长/眼睛到鼻子的距离，此数越大，头的长轴越长（脸越长，已弃用）
# FACE_WIDTH = 1.1  # 脸部的宽度比例，此数越大，头的半径越长
# HEAD_CENTER_HEIGHT = 5  # 头部中心的y轴为鼻子往上调整此数分之一的长轴，此数越大，头整体越往上

# HEIGHT_SCALE = 17.5  # 关节点中心的搜索矩形的高，激光点与中心的距离小于此半径的认为是同一物体
# WIDTH_SCALE = 13  # 关节点中心的搜索矩形的宽，激光点与中心的距离小于此半径的认为是同一物体
# =======================================================
