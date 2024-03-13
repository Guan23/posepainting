#!usr/bin/env python
# _*_ encoding: utf-8 _*_
# Author: Guan
# Create Time: 2021/12/15 上午10:53

import os
import time
import sys

import numpy as np

from load_data import *
from project import main
import myconfig as mcfg


# 计算两框的交并比，bbox的格式为x1,y1,x2,y2
def compute_iou(bbox1, bbox2):
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    area_sum = area1 + area2
    # 获得交集框的左上和右下坐标
    left_line = max(bbox1[0], bbox2[0])
    right_line = min(bbox1[2], bbox2[2])
    up_line = max(bbox1[1], bbox2[1])
    bottom_line = min(bbox1[3], bbox2[3])
    if left_line >= right_line or up_line >= bottom_line:
        return 0
    intersect_area = (right_line - left_line) * (bottom_line - up_line)
    return intersect_area / (area_sum - intersect_area)


# 判断两个矩形框的位置关系，如果box1包含box2，返回1，反之返回-1，如果二者之间没有包含关系，返回0
def is_involved(bbox1, bbox2):
    # 判断包含关系除了看四点坐标，还要看面积的比例
    width1 = bbox1[2] - bbox1[0]
    height1 = bbox1[3] - bbox1[1]
    width2 = bbox2[2] - bbox2[0]
    height2 = bbox2[3] - bbox2[1]
    area1 = width1 * height1
    area2 = width2 * height2
    # 大框的x1应该比小框的x1小，即小框的x1-大框的x1应该大于0,这里我设置了一个略小于0的变量thres * 大框的宽，即允许有小部分误差存在
    if bbox2[0] - bbox1[0] > mcfg.INVOLVE_WIDTH_THRES * width1 \
            and bbox2[1] - bbox1[1] > mcfg.INVOLVE_HEIGHT_THRES * height1 \
            and bbox1[2] - bbox2[2] > mcfg.INVOLVE_WIDTH_THRES * width1 \
            and bbox1[3] - bbox2[3] > mcfg.INVOLVE_HEIGHT_THRES * height1:
        # 此时要看面积比例，大框面积/小框面积应该 > 1，这里我也设置了一个略小于1的常量，即也允许有误差
        return 1 if area1 / area2 > mcfg.INVOLVE_AREA_THRES else 0

    elif bbox1[0] - bbox2[0] > mcfg.INVOLVE_WIDTH_THRES * width2 \
            and bbox1[1] - bbox2[1] > mcfg.INVOLVE_HEIGHT_THRES * height2 \
            and bbox2[2] - bbox1[2] > mcfg.INVOLVE_WIDTH_THRES * width2 \
            and bbox2[3] - bbox1[3] > mcfg.INVOLVE_HEIGHT_THRES * height2:
        return -1 if area2 / area1 > mcfg.INVOLVE_AREA_THRES else 0
    else:
        return 0


# 根据预测的3d关节点，给出预测的人体位置
def get_pred_loc(label, Loc2Velo, joint_points_3d_list, i):
    loc = np.append(label.loc, 1)  # 添加反射强度一维
    gt_loc3d = Loc2Velo.dot(loc.transpose()).transpose()  # 把cam坐标系下的loc转到激光坐标系下

    # 选取躯干点的位置
    shoulder_x = np.average(joint_points_3d_list[i, 5:7, 0])
    hip_x = np.average(joint_points_3d_list[i, 11:13, 0])
    shoulder_y = np.average(joint_points_3d_list[i, 5:7, 1])
    hip_y = np.average(joint_points_3d_list[i, 11:13, 1])

    # 求出行人当前位置
    # pred_x = np.median(joint_points_3d_list[i, :, 0]) + mcfg.BODY_DEPTH  # 激光雷达打的是表面，人体厚度大约是7.5cm
    pred_x = np.average((shoulder_x, hip_x)) + mcfg.BODY_DEPTH  # 激光雷达打的是表面，人体厚度大约是7.5cm
    pred_y = np.average((shoulder_y, hip_y))

    pred_z = gt_loc3d[2]  # z轴直接取label的z，不引入此维度的误差
    pred_i = 1  # 反射强度都设置1,不引入此维度的误差
    pred_loc3d = np.array([pred_x, pred_y, pred_z, pred_i])
    if mcfg.DETAILS:
        print("--- pred_loc3d: {}, gt_loc3d: {} ---".format(pred_loc3d, gt_loc3d))
    return gt_loc3d, pred_loc3d


# 评估函数，对一帧数据进行精度评估，只评估loc中深度维的ale，注意此loc指的是底面中心，而非立体框空间中心
def evaluation(idx, pred_boxes, joint_points_3d_list):
    if pred_boxes is None:
        return [0, 0, 0, 0], [0, 0, 0], [0, 0, 0], [-1, -1, -1]
    gt_label_lists = get_label(idx)  # 每个列表元素均为一个Object3d对象
    calib = get_calib_fromfile(idx)  # 获得各个矩阵
    R0_rect = calib['R0_rect']
    Tr_velo2cam = calib['Tr_velo2cam']
    Loc2Velo = np.linalg.inv(R0_rect.dot(Tr_velo2cam))
    loc_list_easy = []
    loc_list_moderate = []
    loc_list_hard = []
    num_2d_easy = 0
    num_2d_moderate = 0
    num_2d_hard = 0
    num_gt_total = 0
    num_gt_easy = 0
    num_gt_moderate = 0
    num_gt_hard = 0

    # 对所有预测框进行评估
    for i, pred_box in enumerate(pred_boxes):
        # 可能出现有2d的目标检测框但是没有3d关节点
        if i >= len(joint_points_3d_list):
            print("2D have box but 3d human no pose!!!!")
            continue
        num_gt_total = 0
        num_gt_easy = 0
        num_gt_moderate = 0
        num_gt_hard = 0
        pred_box = np.reshape(pred_box, -1)
        # 对所有gt_labels进行遍历，计算预测框和真实框的IOU，超过阈值就认为是同一个人体
        for label in gt_label_lists:
            if label.cls_id == 1:  # pedestrian:1, Person_sitting:2, Cyclist:3
                num_gt_total += 1
                if label.level == 0:  # easy:0, moderate:1, hard:2
                    num_gt_easy += 1
                    # 除了用iou判断是否是那个人外，还要考虑遮挡严重的情况，此时预测框是真实框的一部分，被真实框包含在里面
                    if compute_iou(pred_box, label.box2d) > mcfg.EASY_IOU_THRESH \
                            or is_involved(pred_box, label.box2d) == -1:
                        num_2d_easy += 1
                        loc3d, pred_loc_easy = get_pred_loc(label, Loc2Velo, joint_points_3d_list, i)
                        loc_list_easy.append([loc3d, pred_loc_easy])
                        continue
                elif label.level == 1:  # moderate
                    num_gt_moderate += 1
                    # 除了用iou判断是否是那个人外，还要考虑遮挡严重的情况，此时预测框是真实框的一部分，被真实框包含在里面
                    if compute_iou(pred_box, label.box2d) > mcfg.MODERATE_IOU_THRESH \
                            or is_involved(pred_box, label.box2d) == -1:
                        num_2d_moderate += 1
                        loc3d, pred_loc_moderate = get_pred_loc(label, Loc2Velo, joint_points_3d_list, i)
                        loc_list_moderate.append([loc3d, pred_loc_moderate])
                        continue
                elif label.level == 2:  # hard
                    num_gt_hard += 1
                    # 除了用iou判断是否是那个人外，还要考虑遮挡严重的情况，此时预测框是真实框的一部分，被真实框包含在里面
                    if compute_iou(pred_box, label.box2d) > mcfg.HARD_IOU_THRESH or \
                            is_involved(pred_box, label.box2d) == -1:
                        num_2d_hard += 1
                        loc3d, pred_loc_hard = get_pred_loc(label, Loc2Velo, joint_points_3d_list, i)
                        loc_list_hard.append([loc3d, pred_loc_hard])
                        continue
                else:
                    pass
    # 可能三种难度模式都没有预测出人体姿态
    if len(loc_list_easy) == 0 and len(loc_list_moderate) == 0 and len(loc_list_hard) == 0:
        print("\n---------------There is no human bbox!!!---------------\n")
        return [0, 0, 0, 0], [0, 0, 0], [0, 0, 0], [-1, -1, -1]
    loc_numpy_easy = np.array(loc_list_easy)
    loc_numpy_moderate = np.array(loc_list_moderate)
    loc_numpy_hard = np.array(loc_list_hard)

    num_3d_easy = len(loc_numpy_easy)
    num_3d_moderate = len(loc_numpy_moderate)
    num_3d_hard = len(loc_numpy_hard)

    # 每个人预测的x'y'z'与实际的xyz求距离误差，即sqrt((x'-x)^2 + (y'-y)^2 + (z'-z)^2)
    ale_easy = 0
    ale_moderate = 0
    ale_hard = 0

    if len(loc_numpy_easy) > 0:
        ale_easy = np.sqrt(np.sum(np.power((loc_numpy_easy[:, 0] - loc_numpy_easy[:, 1]), 2), 1))
    if len(loc_numpy_moderate) > 0:
        ale_moderate = np.sqrt(np.sum(np.power((loc_numpy_moderate[:, 0] - loc_numpy_moderate[:, 1]), 2), 1))
    if len(loc_numpy_hard) > 0:
        ale_hard = np.sqrt(np.sum(np.power((loc_numpy_hard[:, 0] - loc_numpy_hard[:, 1]), 2), 1))
    nums_gt = [num_gt_total, num_gt_easy, num_gt_moderate, num_gt_hard]
    nums_2d = [num_2d_easy, num_2d_moderate, num_2d_hard]
    nums_3d = [num_3d_easy, num_3d_moderate, num_3d_hard]
    ales = np.array([ale_easy, ale_moderate, ale_hard], dtype=object)

    if mcfg.DETAILS:
        print("---------------total real people nums:{}---------------".format(num_gt_total))
        print("---------------2d people nums:{}---------------".format(num_2d_easy + num_2d_moderate + num_2d_hard))
        print("---------------3d people nums:{}---------------".format(num_3d_easy + num_3d_moderate + num_3d_hard))
    return nums_gt, nums_2d, nums_3d, ales


# 计算各个难度的ALA，以及全体数据的ALA
def compute_ala(ale_numpy_easy, ale_numpy_moderate, ale_numpy_hard):
    ale_easy_l1 = ale_numpy_easy[ale_numpy_easy < mcfg.ALA_THRESH[0]]
    ale_easy_l2 = ale_numpy_easy[ale_numpy_easy < mcfg.ALA_THRESH[1]]
    ale_easy_l3 = ale_numpy_easy[ale_numpy_easy < mcfg.ALA_THRESH[2]]

    ale_moderate_l1 = ale_numpy_moderate[ale_numpy_moderate < mcfg.ALA_THRESH[0]]
    ale_moderate_l2 = ale_numpy_moderate[ale_numpy_moderate < mcfg.ALA_THRESH[1]]
    ale_moderate_l3 = ale_numpy_moderate[ale_numpy_moderate < mcfg.ALA_THRESH[2]]

    ale_hard_l1 = ale_numpy_hard[ale_numpy_hard < mcfg.ALA_THRESH[0]]
    ale_hard_l2 = ale_numpy_hard[ale_numpy_hard < mcfg.ALA_THRESH[1]]
    ale_hard_l3 = ale_numpy_hard[ale_numpy_hard < mcfg.ALA_THRESH[2]]

    easy_ala = np.array([len(ale_easy_l1), len(ale_easy_l2), len(ale_easy_l3)])
    moderate_ala = np.array([len(ale_moderate_l1), len(ale_moderate_l2), len(ale_moderate_l3)])
    hard_ala = np.array([len(ale_hard_l1), len(ale_hard_l2), len(ale_hard_l3)])
    total_ala = easy_ala + moderate_ala + hard_ala

    return easy_ala, moderate_ala, hard_ala, total_ala


# 评估全体数据
def total_eval(maxnum=1779):
    ale_list_easy = []
    ale_list_moderate = []
    ale_list_hard = []
    idx_list = []
    fn2d_list = []
    fn3d_list = []
    bad_list = []
    ped_num = [0 for _ in range(10)]  # 初始化10个0

    filenames = os.listdir(os.path.join(mcfg.ROOT_SPLIT_PATH, "image_2"))
    filenames.sort()
    for i, idx in enumerate(filenames):
        if i >= maxnum:
            break
        idx = idx.split(".")[0]
        bad_idx = False
        print("第{}帧数据，编号为:{}".format(i + 1, idx))
        pred_boxes, joint_points_3d_list = main(idx)
        if pred_boxes is None:
            continue
        nums_gt, nums_2d, nums_3d, ales = evaluation(idx, pred_boxes, joint_points_3d_list)
        # good_error = 0.03
        bad_error = 1.0

        # 误差列表和对应的idx列表
        if type(ales[0]) == np.ndarray:
            for ale0 in ales[0]:
                ale_list_easy.append(ale0)
                if ale0 > bad_error:
                    bad_idx = True
        if type(ales[1]) == np.ndarray:
            for ale1 in ales[1]:
                ale_list_moderate.append(ale1)
                if ale1 > bad_error:
                    bad_idx = True
        if type(ales[2]) == np.ndarray:
            for ale2 in ales[2]:
                ale_list_hard.append(ale2)
                if ale2 > bad_error:
                    bad_idx = True
        idx_list.append(idx)
        if bad_idx:
            bad_list.append(idx)
        # 2d和3d分别漏检的帧idx
        if nums_gt[0] > sum(nums_2d):
            fn2d_list.append(idx)
        if sum(nums_2d) > sum(nums_3d):
            fn3d_list.append(idx)
        # 统计总共的漏检率
        ped_num[0] += nums_gt[0]
        ped_num[1] += nums_gt[1]
        ped_num[2] += nums_gt[2]
        ped_num[3] += nums_gt[3]

        ped_num[4] += nums_2d[0]
        ped_num[5] += nums_2d[1]
        ped_num[6] += nums_2d[2]

        # ped_num[7] += nums_3d[0]
        # ped_num[8] += nums_3d[1]
        # ped_num[9] += nums_3d[2]
    # 总结部分

    ale_numpy_easy = np.array(ale_list_easy)
    ale_numpy_moderate = np.array(ale_list_moderate)
    ale_numpy_hard = np.array(ale_list_hard)
    idx_numpy = np.array(idx_list)

    easy_ped_num_3d = len(ale_numpy_easy)
    moderate_ped_num_3d = len(ale_numpy_moderate)
    hard_ped_num_3d = len(ale_numpy_hard)
    total_ped_num_3d = easy_ped_num_3d + moderate_ped_num_3d + hard_ped_num_3d
    # print(idx_numpy)
    # print(ale_numpy_easy)
    # print(ale_numpy_moderate)
    # print(ale_numpy_hard)

    easy_ala, moderate_ala, hard_ala, total_ala = compute_ala(ale_numpy_easy, ale_numpy_moderate, ale_numpy_hard)

    ale_easy = np.average(ale_numpy_easy)
    ale_moderate = np.average(ale_numpy_moderate)
    ale_hard = np.average(ale_numpy_hard)
    # miss_idx_easy = idx_numpy[ale_numpy_easy == -1]
    # miss_idx_moderate = idx_numpy[ale_numpy_moderate == -1]
    # miss_idx_hard = idx_numpy[ale_numpy_hard == -1]
    # bad_eg = ale_numpy_easy[ale_numpy_easy > bad_error]
    # bad_idx = idx_numpy[ale_numpy_hard > bad_error]
    # good_eg = ale_numpy_easy[(ale_numpy_easy < good_error) & (ale_numpy_easy != -1)]
    # good_idx = idx_numpy[(ale_numpy_easy < good_error) & (ale_numpy_easy != -1)]

    # 存储日志文件
    if mcfg.SAVE_LOG:
        log_folder = os.path.join(mcfg.LOG_DIR, time.strftime("%Y_%m_%d", time.localtime()))
        os.makedirs(log_folder, exist_ok=True)
        log_file = os.path.join(log_folder, time.strftime("%H_%M_%S", time.localtime()) + ".txt")
        with open(log_file, "w+") as f:
            f.write("-----------------------------------------------------------\n")
            f.write("\n目标检测使用的网络为: {}\n".format(mcfg.DETECT_MODEL))
            f.write("KITTI数据集所有包含行人类的数据帧数量为: {}\n".format(mcfg.TOTAL_FRAME))
            f.write("KITTI数据集所有行人样本数量为: {}\n".format(mcfg.TOTAL_PED))
            f.write("KITTI数据集所有简单行人样本数量为: {}\n".format(mcfg.EASY_PED))
            f.write("KITTI数据集所有中等行人样本数量为: {}\n".format(mcfg.MODERATE_PED))
            f.write("KITTI数据集所有困难行人样本数量为: {}\n".format(mcfg.HARD_PED))

            f.write("-----------------------------------------------------------\n")
            f.write("\n本次检测的数据帧数量: {}，目标检测得到的行人数量为: {}\n".format(maxnum, int(ped_num[0])))
            f.write("隶属于三种难度模式的框数量为: {}，未知难度的框数量为: {}\n".
                    format(int(ped_num[1] + ped_num[2] + ped_num[3]),
                           int(ped_num[0] - ped_num[1] - ped_num[2] - ped_num[3])))

            f.write("-----------------------------------------------------------\n")
            f.write("\nEasy模式，可以得到2d检测框的行人数量: {}\n".format(int(ped_num[1])))
            f.write("可以得到2d姿态估计的行人数量: {}，可以得到3d姿态估计的行人数量: {}\n"
                    .format(int(ped_num[4]), easy_ped_num_3d))
            f.write("2d目标检测的漏检率: {:.3f}%，2d姿态估计的漏检率: {:.3f}%，3d姿态估的漏检率: {:.3f}%\n"
                    .format((mcfg.EASY_PED - ped_num[1]) / mcfg.EASY_PED * 100,
                            (ped_num[1] - ped_num[4]) / ped_num[1] * 100,
                            (ped_num[1] - easy_ped_num_3d) / ped_num[1] * 100))
            f.write("平均定位误差ALE: {:.3f}m\n".format(ale_easy))
            f.write("平均定位精度ALA: <0.5m: {}个，占比{:.3f}%, <1.0m: {}个，占比{:.3f}%, <2.0m: {}个，占比{:.3f}%\n"
                    .format(easy_ala[0],
                            easy_ala[0] / easy_ped_num_3d * 100,
                            easy_ala[1],
                            easy_ala[1] / easy_ped_num_3d * 100,
                            easy_ala[2],
                            easy_ala[2] / easy_ped_num_3d * 100))
            # f.write("\n漏检的idx: {}\n\n".format(miss_idx_easy))

            f.write("\n-----------------------------------------------------------\n")
            f.write("Moderate模式，可以得到2d检测框的行人数量: {}\n".format(int(ped_num[2])))
            f.write("可以得到2d姿态估计的行人数量: {}，可以得到3d姿态估计的行人数量: {}\n"
                    .format(int(ped_num[5]), moderate_ped_num_3d))
            f.write("2d目标检测的漏检率: {:.3f}%，2d姿态估计的漏检率: {:.3f}%，3d姿态估计的漏检率: {:.3f}%\n"
                    .format((mcfg.MODERATE_PED - ped_num[2]) / mcfg.MODERATE_PED * 100,
                            (ped_num[2] - ped_num[5]) / ped_num[2] * 100,
                            (ped_num[2] - moderate_ped_num_3d) / ped_num[2] * 100))
            f.write("平均定位误差ALE: {:.3f}m\n".format(ale_moderate))
            f.write("平均定位精度ALA: <0.5m: {}个，占比{:.3f}%, <1.0m: {}个，占比{:.3f}%, <2.0m: {}个，占比{:.3f}%\n"
                    .format(moderate_ala[0],
                            moderate_ala[0] / moderate_ped_num_3d * 100,
                            moderate_ala[1],
                            moderate_ala[1] / moderate_ped_num_3d * 100,
                            moderate_ala[2],
                            moderate_ala[2] / moderate_ped_num_3d * 100))
            # f.write("\n漏检的idx: {}\n\n".format(miss_idx_moderate))

            f.write("\n-----------------------------------------------------------\n")
            f.write("Hard模式，可以得到2d检测框的行人数量: {}\n".format(int(ped_num[3])))
            f.write("可以得到2d姿态估计的行人数量: {}，可以得到3d姿态估计的行人数量: {}\n"
                    .format(int(ped_num[6]), hard_ped_num_3d))
            f.write("2d目标检测的漏检率: {:.3f}%，2d姿态估计的漏检率: {:.3f}%，3d姿态估计的漏检率: {:.3f}%\n"
                    .format((mcfg.HARD_PED - ped_num[3]) / mcfg.HARD_PED * 100,
                            (ped_num[3] - ped_num[6]) / ped_num[3] * 100,
                            (ped_num[3] - hard_ped_num_3d) / ped_num[3] * 100))
            f.write("平均定位误差ALE: {:.3f}m\n".format(ale_hard))
            f.write("平均定位精度ALA: <0.5m: {}个，占比{:.3f}%, <1.0m: {}个，占比{:.3f}%, <2.0m: {}个，占比{:.3f}%\n"
                    .format(hard_ala[0],
                            hard_ala[0] / hard_ped_num_3d * 100,
                            hard_ala[1],
                            hard_ala[1] / hard_ped_num_3d * 100,
                            hard_ala[2],
                            hard_ala[2] / hard_ped_num_3d * 100))
            # f.write("\n漏检的idx: {}\n\n".format(miss_idx_hard))

            f.write("\n-----------------------------------------------------------\n")
            f.write("\n全体数据的平均定位精度ALA: <0.5m: {}个，占比{:.3f}%, <1.0m: {}个，占比{:.3f}%, <2.0m: {}个，占比{:.3f}%\n"
                    .format(total_ala[0],
                            total_ala[0] / total_ped_num_3d * 100,
                            total_ala[1],
                            total_ala[1] / total_ped_num_3d * 100,
                            total_ala[2],
                            total_ala[2] / total_ped_num_3d * 100))
            f.write("\n2d姿态漏检的idx: {}\n\n".format(fn2d_list))
            f.write("\n2d未漏但3d姿态漏检的idx: {}\n\n".format(fn3d_list))
            f.write("\n较差的帧: {}\n".format(bad_list))
            # f.write("较差(>2.5m)的帧数: {}，较差的示例如下:\n数据帧idx:\n{}\n误差error:\n{}\n\n"
            #         .format(len(bad_eg), bad_idx, bad_eg))
            # f.write("较好(<0.03m)的帧数: {}，较好的示例如下:\n数据帧idx:\n{}\n误差error:\n{}\n"
            #         .format(len(good_eg), good_idx, good_eg))
            f.write("\n-----------------------------------------------------------\n")


if __name__ == '__main__':
    print('\n-----------------------start-----------------------\n')

    # 批量评估
    # total_eval(maxnum=1779)

    # 单张测试
    idx = "006224"
    pred_boxes, joint_points_3d_list = main(idx)
    num_gt, num_2d, num_3d, ales = evaluation(idx, pred_boxes, joint_points_3d_list)
    print("\n------extra------\n")
    print(num_gt)
    print(num_2d)
    print(num_3d)
    print(ales)

    print('\n------------------------end------------------------\n')
