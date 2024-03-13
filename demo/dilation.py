# _*_ coding utf-8 _*_
# developer: Guan
# time: 2021/11/22-12:24

import cv2
import time
import os
import copy

import numpy as np

import myconfig as mcfg

# COCO_KEYPOINT_INDEXES = {
#     0: 'nose',
#     1: 'left_eye',
#     2: 'right_eye',
#     3: 'left_ear',
#     4: 'right_ear',
#     5: 'left_shoulder',
#     6: 'right_shoulder',
#     7: 'left_elbow',
#     8: 'right_elbow',
#     9: 'left_wrist',
#     10: 'right_wrist',
#     11: 'left_hip',
#     12: 'right_hip',
#     13: 'left_knee',
#     14: 'right_knee',
#     15: 'left_ankle',
#     16: 'right_ankle'
# }

# 每个身体部位的颜色，可视化用
COLOR_ = (
    (255, 228, 225), (100, 149, 237),
    (64, 224, 208), (152, 251, 152), (107, 142, 35), (205, 92, 92),
    (255, 140, 0), (199, 21, 133), (148, 0, 211), (255, 250, 205)
)

# 每个身体部位的掩码
CHANNEL_VAL_ = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

# pose_pred_test = [
#     [88, 29], [82, 26], [94, 26], [77, 30], [99, 30],
#     [81, 43], [95, 43], [70, 56], [106, 56], [60, 66], [116, 66],
#     [81, 64], [95, 64], [78, 83], [98, 83], [74, 98], [102, 98]
# ]

pose_pred_test = [
    [75, 28], [69, 26], [79, 25], [61, 30], [86, 29],
    [46, 57], [101, 55], [28, 98], [120, 98], [46, 125], [97, 126],
    [57, 141], [92, 140], [64, 205], [82, 202], [67, 263], [81, 261]
]


# 获取头部和躯干的朝向，0代表正向和反向，1代表朝左，2代表朝右
# TODO：可以改为读取label中的orientation信息，那么此模块就需要3D目标检测的输出了
# 已弃用
def get_orientation(pose_pred):
    # 头部朝向和躯干朝向，0代表正面或反面，1代表朝左，2代表朝右
    head_orientation = 0
    torso_orientation = 0
    # 根据鼻子与两耳的x坐标判断头的朝向
    if pose_pred[0][0] < pose_pred[3][0] and pose_pred[0][0] < pose_pred[4][0]:
        head_orientation = 1
    elif pose_pred[0][0] > pose_pred[3][0] and pose_pred[0][0] > pose_pred[4][0]:
        head_orientation = 2
    # 计算躯干的长度与宽度的比例，超过阈值则认为是侧身，躯干的朝向跟头部一致
    torso_width = abs(pose_pred[6][0] - pose_pred[5][0]) + abs(pose_pred[6][0] - pose_pred[5][0])
    torso_length = pose_pred[11][1] - pose_pred[5][1] + pose_pred[12][1] - pose_pred[6][1]
    torso_width = max(1, torso_width)
    torso_length = max(1, torso_length)
    if torso_length / torso_width > 2:
        torso_orientation = head_orientation
    return head_orientation, torso_orientation


# pose_pred shape:(17, 2)
def get_human_proportion(pose_pred, box_size=None):
    if not isinstance(pose_pred, np.ndarray):
        pose_pred = np.array(pose_pred)
    # width, height = box_size
    # head_orientation, torso_orientation = get_orientation(pose_pred)
    # print("--------head_orientation:{}, torso_orientation:{}--------\n".format(head_orientation, torso_orientation))
    # 身体躯干点（从肩膀到髋部）用多边形表示，需要按顺序的四个点
    # TODO：后期改成用向量叉乘来得到四个点的顺时针/逆时针顺序
    if pose_pred[5][0] < pose_pred[6][0]:
        torso_left_top = pose_pred[5]
        torso_right_top = pose_pred[6]
    else:
        torso_left_top = pose_pred[6]
        torso_right_top = pose_pred[5]
    if pose_pred[11][0] < pose_pred[12][0]:
        torso_left_bottom = pose_pred[11]
        torso_right_bottom = pose_pred[12]
    else:
        torso_left_bottom = pose_pred[12]
        torso_right_bottom = pose_pred[11]

    # 这里得到的躯干高度是正确的，但躯干宽度可能由于侧身等原因是不正确的
    torso_width = np.mean((np.linalg.norm(torso_right_top - torso_left_top),
                           np.linalg.norm(torso_right_bottom - torso_left_bottom)))

    torso_height = np.mean((np.linalg.norm(torso_left_top - torso_left_bottom),
                            np.linalg.norm(torso_right_top - torso_right_bottom)))
    baseline = torso_height / mcfg.BODY_HEIGHT  # 基线长度（即头部直径）为躯干高度除以3
    # 正常人躯干宽度等于1.6个头，将这个值与得到的虚躯干宽度求平均，就比较接近真实的躯干宽度了
    new_torso_width = (mcfg.BODY_WIDTH * baseline + torso_width) / 2.0
    offset = (new_torso_width - torso_width) / 2.0
    torso_left_top[0] -= offset
    torso_left_bottom[0] -= offset
    torso_right_top[0] += offset
    torso_right_bottom[0] += offset

    # 头部拟合成圆形，圆心x为双耳的x取平均，圆心y为双眼的y与鼻子的y取平均，半径为基线长度
    head_center_x = (pose_pred[3][0] + pose_pred[4][0]) / 2.0
    head_center_y = ((pose_pred[1][1] + pose_pred[2][1]) / 2.0 + pose_pred[0][1]) / 2.0
    head_radius = round(baseline * mcfg.HEAD_RATIO)
    head_center_points = (round(head_center_x), round(head_center_y))

    # 大臂、小臂，大腿、小腿的长度
    big_arm_length = (np.linalg.norm(pose_pred[5] - pose_pred[7]) +
                      np.linalg.norm(pose_pred[6] - pose_pred[8])) / 2
    small_arm_length = (np.linalg.norm(pose_pred[7] - pose_pred[9]) +
                        np.linalg.norm(pose_pred[8] - pose_pred[10])) / 2
    big_leg_length = (np.linalg.norm(pose_pred[11] - pose_pred[13]) +
                      np.linalg.norm(pose_pred[12] - pose_pred[14])) / 2
    small_leg_length = (np.linalg.norm(pose_pred[13] - pose_pred[15]) +
                        np.linalg.norm(pose_pred[14] - pose_pred[16])) / 2

    # 手臂、腿的宽度
    big_arm_thk = max(big_arm_length / mcfg.BIG_ARM_RATIO, 1)
    small_arm_thk = max(small_arm_length / mcfg.SMALL_ARM_RATIO, 1)
    big_leg_thk = max(big_leg_length / mcfg.BIG_LEG_RATIO, 1)
    small_leg_thk = max(small_leg_length / mcfg.SMALL_LEG_RATIO, 1)

    human_proportion = [head_center_points, head_radius,
                        torso_left_top, torso_right_top, torso_left_bottom, torso_right_bottom,
                        big_arm_thk, small_arm_thk, big_leg_thk, small_leg_thk]

    return human_proportion


# 绘制人体骨架轮廓，把人体分成了10个部分，头、躯干、左大/小臂，右大/小臂，左大/小腿，右大/小腿
# TODO:后绘制的图形会覆盖掉先绘制的图形，所以最好确定人体的方向，以确定先画左侧还是右侧
def draw_outline(image, pose_pred, color, human_proportion):
    # image_outline = image.copy()
    head_center_points, head_radius, \
        torso_left_top, torso_right_top, torso_left_bottom, torso_right_bottom, \
        big_arm_thk, small_arm_thk, big_leg_thk, small_leg_thk = human_proportion
    # draw outlines
    head_outline = cv2.circle(image, head_center_points, head_radius, color[0], -1)
    # head_outline = cv2.ellipse(image, head_center_points, (head_long_axis, head_short_axis), 90, 0,
    #                            360, color[0], -1)  # head
    pts = np.array([torso_left_top, torso_right_top,
                    torso_right_bottom, torso_left_bottom], dtype=np.int32).reshape((-1, 1, 2))

    torso_outline = cv2.fillPoly(head_outline, [pts], color[1])  # torso
    # torso_outline = cv2.rectangle(head_outline, torso_left_top, torso_right_bottom, color[1], -1)  # torso

    left_arms_outline = cv2.line(torso_outline, tuple(pose_pred[7]), tuple(pose_pred[9]), color[3],
                                 int(small_arm_thk))  # left small arm
    left_arms_outline = cv2.line(left_arms_outline, tuple(pose_pred[5]), tuple(pose_pred[7]), color[2],
                                 int(big_arm_thk))  # left big arm

    right_arms_outline = cv2.line(left_arms_outline, tuple(pose_pred[8]), tuple(pose_pred[10]), color[5],
                                  int(small_arm_thk))  # right small arm
    right_arms_outline = cv2.line(right_arms_outline, tuple(pose_pred[6]), tuple(pose_pred[8]), color[4],
                                  int(big_arm_thk))  # right big arm

    left_legs_outline = cv2.line(right_arms_outline, tuple(pose_pred[13]), tuple(pose_pred[15]), color[7],
                                 int(small_leg_thk))  # left small leg
    left_legs_outline = cv2.line(left_legs_outline, tuple(pose_pred[11]), tuple(pose_pred[13]), color[6],
                                 int(big_leg_thk))  # left big leg

    right_legs_outline = cv2.line(left_legs_outline, tuple(pose_pred[14]), tuple(pose_pred[16]), color[9],
                                  int(small_leg_thk))  # right small leg
    one_man_outline = cv2.line(right_legs_outline, tuple(pose_pred[12]), tuple(pose_pred[14]), color[8],
                               int(big_leg_thk))  # right big leg

    return one_man_outline


if __name__ == "__main__":
    print("\n----------start----------\n")
    human_size = get_human_proportion(pose_pred_test)
    image = cv2.imread("p1.jpg", flags=1)
    print(image.shape)
    image_outline = draw_outline(image, pose_pred_test, COLOR_, human_size)
    print(image_outline.shape)
    cv2.namedWindow("res",flags=1)
    cv2.imshow("res", image_outline)
    cv2.waitKey(0)
    cv2.imwrite("./p1s.jpg", image_outline)
    cv2.destroyAllWindows()


    print("\n-----------end-----------\n")
