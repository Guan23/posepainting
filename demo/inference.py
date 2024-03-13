from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import shutil

from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np

import sys

sys.path.append("../lib")
import time

# import _init_paths
import models
from config import cfg
from config import update_config
from core.inference import get_final_preds
from utils.transforms import get_affine_transform

import myconfig as mcfg
from object3d_kitti import get_objects_from_label

__all__ = ["image_flow", "preprocess"]

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


# 生成输出目录
def prepare_output_dirs(prefix='/output/'):
    pose_dir = os.path.join(prefix, "pose")
    if os.path.exists(pose_dir) and os.path.isdir(pose_dir):
        shutil.rmtree(pose_dir)
    os.makedirs(pose_dir, exist_ok=True)
    return pose_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, default='./coco_hrnet_w32_256x192.yaml')
    parser.add_argument('--videoFile', type=str, default='test01.mp4')
    parser.add_argument('--outputDir', type=str, default='./output/')
    parser.add_argument('--inferenceFps', type=int, default=10)
    parser.add_argument('--writeBoxFrames', default=True, action='store_false')  # 是否画目标检测的框

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    # added
    # parser.add_argument('--imageFile', type=str, default='../test_image/t01.png')  # 测试图片
    parser.add_argument('--saveCSV', default=False, action='store_true')  # 是否保存csv表格数据
    args = parser.parse_args()

    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args


# opencv显示图片，按s键保存图片，路径默认为./当前时间.png
def show_image(image, img_file="./output"):
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow("img", image)
    k = cv2.waitKey(0) & 0xff
    if chr(k) == 's' or chr(k) == 'S':
        print("-----------saving pic!-----------")
        current_time = time.localtime()
        save_file = str(time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
        cv2.imwrite(os.path.join(img_file, '{}.png'.format(save_file)), image)
    else:
        print("-----------close without saving pic-----------")
    cv2.destroyAllWindows()


# 绘制2d人体姿态
def draw_pose(image, pose_preds, line_color=(13, 23, 227), line_thk=2):
    for i, coords in enumerate(pose_preds):
        for i, coord in enumerate(coords):
            x, y = int(coord[0]), int(coord[1])
            cv2.circle(image, (x, y), 2, (64, 125, 255), 2)
        cv2.line(image, tuple(coords[0]), tuple(coords[1]), line_color, line_thk)
        cv2.line(image, tuple(coords[0]), tuple(coords[2]), line_color, line_thk)
        cv2.line(image, tuple(coords[1]), tuple(coords[3]), line_color, line_thk)
        cv2.line(image, tuple(coords[2]), tuple(coords[4]), line_color, line_thk)

        cv2.line(image, tuple(coords[5]), tuple(coords[7]), line_color, line_thk)
        cv2.line(image, tuple(coords[7]), tuple(coords[9]), line_color, line_thk)
        cv2.line(image, tuple(coords[6]), tuple(coords[8]), line_color, line_thk)
        cv2.line(image, tuple(coords[8]), tuple(coords[10]), line_color, line_thk)

        cv2.line(image, tuple(coords[11]), tuple(coords[13]), line_color, line_thk)
        cv2.line(image, tuple(coords[13]), tuple(coords[15]), line_color, line_thk)
        cv2.line(image, tuple(coords[12]), tuple(coords[14]), line_color, line_thk)
        cv2.line(image, tuple(coords[14]), tuple(coords[16]), line_color, line_thk)

        cv2.line(image, tuple(coords[5]), tuple(coords[6]), line_color, line_thk)
        cv2.line(image, tuple(coords[11]), tuple(coords[12]), line_color, line_thk)

        upmid = (coords[5] + coords[6]) // 2
        downmid = (coords[11] + coords[12]) // 2

        cv2.line(image, tuple(coords[0]), tuple(upmid), line_color, line_thk)
        cv2.line(image, tuple(upmid), tuple(downmid), line_color, line_thk)
    return image


# 获得人的检测框
def get_person_detection_boxes(model, img, threshold=0.5):
    pil_image = Image.fromarray(img)  # Load the image
    transform = transforms.Compose([transforms.ToTensor()])  # Defing PyTorch Transform
    transformed_img = transform(pil_image)  # Apply the transform to the image
    pred = model([transformed_img.to(CTX)])  # Pass the image to the model
    # Use the first detected person
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].cpu().detach().numpy())]  # Bounding boxes
    pred_scores = list(pred[0]['scores'].cpu().detach().numpy())

    person_boxes = []
    # Select box has score larger than threshold and is person
    for pred_class, pred_box, pred_score in zip(pred_classes, pred_boxes, pred_scores):
        if (pred_score > threshold) and (pred_class == 'person'):
            person_boxes.append(pred_box)

    return person_boxes


def get_pose_estimation_prediction(pose_model, image, centers, scales, transform):
    rotation = 0

    # pose estimation transformation
    model_inputs = []
    for center, scale in zip(centers, scales):
        trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
        # Crop smaller image of people
        model_input = cv2.warpAffine(
            image,
            trans,
            (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR)

        # hwc -> 1chw
        model_input = transform(model_input)  # .unsqueeze(0)
        model_inputs.append(model_input)

    # 如果检测不到人，则返回None
    if len(model_inputs) == 0:
        return None
    # n * 1chw -> nchw
    model_inputs = torch.stack(model_inputs)

    # compute output heatmap
    output = pose_model(model_inputs.to(CTX))
    coords, _ = get_final_preds(
        cfg,
        output.cpu().detach().numpy(),
        np.asarray(centers),
        np.asarray(scales))

    return coords


def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0] - bottom_left_corner[0]
    box_height = top_right_corner[1] - bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


# 图片处理流
def image_flow(args, idx, box_model, pose_model, pose_transform, csv_output_rows, imagePath):
    image_bgr = cv2.imread(imagePath)  # opencv的channel是bgr的顺序
    total_now = time.time()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # bgr转rgb
    # Clone 2 image for person detection and pose estimation
    if cfg.DATASET.COLOR_RGB:
        image_per = image_rgb.copy()
        image_pose = image_rgb.copy()
    else:
        image_per = image_bgr.copy()
        image_pose = image_bgr.copy()

    image_debug = image_bgr.copy()  # Clone 1 image for debugging purpose
    # object detection box
    now = time.time()
    # 使用深度学习获得行人检测框或者直接读取KITTI的训练集的行人框标签
    pred_boxes = get_person_detection_boxes(box_model, image_per, threshold=mcfg.DETECT_SCORE_THRESH)  # 获得人的2D框
    # 直接使用KITTI训练标签中的2D框，草，结果更差
    # label_file = mcfg.ROOT_SPLIT_PATH + 'label/' + ('%s.txt' % idx)
    # gt_label_lists = get_objects_from_label(label_file)
    # pred_boxes = []
    # for i, gt_label in enumerate(gt_label_lists):
    #     if i >= 17:  # 我的电脑显卡最多处理17个HRNet，要不就cuda out of memory了
    #         break
    #     if gt_label.cls_id == 1:  # 只要行人框
    #         left_top = (gt_label.box2d[0], gt_label.box2d[1])
    #         right_down = (gt_label.box2d[2], gt_label.box2d[3])
    #         box_coord = [left_top, right_down]
    #         pred_boxes.append(box_coord)

    # print("pred_boxes: {}".format(pred_boxes))
    # 找不到人的框，直接返回None
    if len(pred_boxes) == 0:
        print("---Could not find person bbox!!!---")
        return None
    then = time.time()
    print("Find person bbox in: {} sec".format(round(then - now, 5)))  # 目标检测的时间消耗
    # 画person类的2D检测框
    if args.writeBoxFrames:
        for i, box in enumerate(pred_boxes):
            cv2.rectangle(image_debug, (int(box[0][0]), int(box[0][1])), (int(box[1][0]), int(box[1][1])),
                          color=(0, 255, 0),
                          thickness=1)  # Draw Rectangle with the coordinates
            cv2.putText(image_debug, str(i + 1), (int(box[0][0]) + 5, int(box[0][1]) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (102, 102, 255), 1)  # 给每个人的框编号
    if mcfg.VISIBLE:
        show_image(image_debug)  # 显示结果图片
    # pose estimation : for multiple people
    centers = []
    scales = []
    for box in pred_boxes:
        # 找到目标检测框的中心和缩放尺度
        center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
        centers.append(center)
        scales.append(scale)
    # 人体关节点检测
    now = time.time()
    pose_preds = get_pose_estimation_prediction(pose_model, image_pose, centers, scales, transform=pose_transform)
    # 找不到人体关节点，也返回None
    if pose_preds.all() is None:
        print("---Could not estimate 2d pose!!!---")
        return None
    then = time.time()
    print("Find person pose in: {} sec".format(round(then - now, 5)))
    new_csv_row = [0, ]  # 图片只有1帧
    # 画2d关节点
    # for coords in pose_preds:
    #     # Draw each point on image
    #     for coord in coords:
    #         x_coord, y_coord = int(coord[0]), int(coord[1])
    #         cv2.circle(image_debug, (x_coord, y_coord), 3, (255, 0, 0), 2)
    #         new_csv_row.extend([x_coord, y_coord])
    total_then = time.time()
    # img_file = os.path.join(pose_dir, 'pose_{}.jpg'.format(total_then))
    # text = "{:03.2f} sec".format(total_then - total_now)  # 总的时间消耗
    # cv2.putText(image_debug, text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    image_debug = draw_pose(image_debug, pose_preds)
    if mcfg.VISIBLE:
        show_image(image_debug)  # 显示结果图片

    # 是否保存csv表格数据
    if args.saveCSV:
        csv_output_rows.append(new_csv_row)
        # write csv
        csv_headers = ['frame']
        for keypoint in COCO_KEYPOINT_INDEXES.values():
            csv_headers.extend([keypoint + '_x', keypoint + '_y'])

        csv_output_filename = os.path.join(args.outputDir, 'pose-data.csv')
        with open(csv_output_filename, 'a+', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(csv_headers)
            csvwriter.writerows(csv_output_rows)
    # print("-------------------OK!-------------------")
    # 输出人体检测2d框(x1,y1,x2,y2)和人体17个关节点坐标(17, 2)
    return pred_boxes, pose_preds


# 预处理函数，在模型预测之前进行的预处理
def preprocess():
    # transformation
    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()
    update_config(cfg, args)
    # pose_dir = prepare_output_dirs(args.outputDir)
    # pose_dir = "output"
    csv_output_rows = []
    if mcfg.DETECT_MODEL == "faster_rcnn":
        box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)  # default:faster-rcnn
    elif mcfg.DETECT_MODEL == "retina_net":
        box_model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
    elif mcfg.DETECT_MODEL == "mask_rcnn":
        box_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    else:
        print("请使用下面三种目标检测网络之一: {}, {}, {}\n".format("faster_rcnn", "retina_net", "mask_rcnn"))
        sys.exit(-1)

    box_model.to(CTX)  # cuda or cpu
    box_model.eval()
    pose_model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        print("=> box_model is: {}".format(mcfg.DETECT_MODEL))
        print('=> loading model from: {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        print('expected model defined in config at TEST.MODEL_FILE')
    pose_model.to(CTX)
    pose_model.eval()
    return args, box_model, pose_model, pose_transform, csv_output_rows


if __name__ == '__main__':
    imagePath = '../test_image/t01.png'
    preprocess(imagePath)
