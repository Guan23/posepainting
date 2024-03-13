import numpy as np
import torch

import mayavi.mlab as mlab
import matplotlib.pyplot as plt
from open3d import *

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


# 点的可视化函数
def visualize_pts(pts, fig=None, bgcolor=(0.8, 0.8, 0.8), fgcolor=(0.0, 0.0, 0.0),
                  show_intensity=True, size=(600, 600), draw_origin=True, point_size=0.08):
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=fgcolor, engine=None, size=size)

    # 通过第四维intensity来赋不同的颜色
    if show_intensity:
        # G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3:], mode='sphere',
        #   colormap='gnuplot', scale_factor=1, figure=fig)
        rgba = np.zeros((pts.shape[0], 4), dtype=np.uint8)
        rgba[:, -1] = 255  # no transparency，点的透明度，0为全透明，255为不透明
        # 赋予颜色
        rgba[:, 0] = pts[:, -1] * 175
        rgba[:, 1] = pts[:, -1] * 254
        rgba[:, 2] = pts[:, -1] * 254
        # print(rgba)
        pts = mlab.pipeline.scalar_scatter(pts[:, 0], pts[:, 1], pts[:, 2])  # plot the points
        pts.add_attribute(rgba, 'colors')  # assign the colors to each point
        pts.data.point_data.set_active_scalars('colors')
        g = mlab.pipeline.glyph(pts)
        g.glyph.glyph.scale_factor = point_size  # set scaling for all the points
        g.glyph.scale_mode = 'data_scaling_off'  # make all the points same size
    else:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
    if draw_origin:
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
        mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), tube_radius=0.1)

    return fig


def draw_grid(x1, y1, x2, y2, fig, tube_radius=None, color=(0.5, 0.5, 0.5)):
    mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    return fig


def draw_multi_grid_range(fig, grid_size=20, bv_range=(-60, -60, 60, 60)):
    for x in range(bv_range[0], bv_range[2], grid_size):
        for y in range(bv_range[1], bv_range[3], grid_size):
            fig = draw_grid(x, y, x + grid_size, y + grid_size, fig)

    return fig


# 绘制函数
def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_scores=None, ref_labels=None, point_size=0.08):
    # 先是把所有要显示的数据转换成numpy的格式
    if not isinstance(points, np.ndarray):
        points = points.cpu().numpy()
    if ref_boxes is not None and not isinstance(ref_boxes, np.ndarray):
        ref_boxes = ref_boxes.cpu().numpy()
    if gt_boxes is not None and not isinstance(gt_boxes, np.ndarray):
        gt_boxes = gt_boxes.cpu().numpy()
    if ref_scores is not None and not isinstance(ref_scores, np.ndarray):
        ref_scores = ref_scores.cpu().numpy()
    if ref_labels is not None and not isinstance(ref_labels, np.ndarray):
        ref_labels = ref_labels.cpu().numpy()

    fig = visualize_pts(points, show_intensity=True, point_size=point_size)

    mlab.view(azimuth=-179, elevation=83.0, distance=45.0, roll=90.0)
    return fig


# def viz_matplot(points):
#     x = points[:, 0]  # x position of point
#     y = points[:, 1]  # y position of point
#     z = points[:, 2]  # z position of point
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(x,  # x
#                y,  # y
#                z,  # z
#                c=z,  # height data for color
#                cmap='rainbow',
#                marker=".")
#     ax.axis()
#     plt.show()


def viz_open3d(points):
    point_cloud = PointCloud()
    point_cloud.points = Vector3dVector(points)
    draw_geometries_with_editing([point_cloud])

# 下面是单独显示3D人体姿态的代码，用这个进行可视化
# h36m骨架连接顺序，每个骨架三个维度，分别为：起始关节，终止关节，左右关节标识(1 left 0 right)
'''
human36m_connectivity_dict = [[0, 1, 0], [1, 2, 0], [2, 6, 0], [5, 4, 1], [4, 3, 1], [3, 6, 1], [6, 7, 0],
                              [7, 8, 0], [8, 16, 0], [9, 16, 0],
                              [8, 12, 0], [11, 12, 0], [10, 11, 0], [8, 13, 1], [13, 14, 1], [14, 15, 1]]
'''
coco_connectivity_dict = [[0, 1], [0, 2], [1, 3], [2, 4],
                          [5, 7], [7, 9], [6, 8], [8, 10],
                          [11, 13], [13, 15], [12, 14], [14, 16],
                          [5, 6], [11, 12],
                          [0, 17], [17, 18]]


def draw3Dpose(pose_3d, ax, color="#3498db", add_labels=False):  # blue, orange
    for i, p in enumerate(coco_connectivity_dict):
        x, y, z = [np.array([pose_3d[p[0], j], pose_3d[p[1], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, c=color)

    RADIUS = 0.750  # space around the subject
    xroot, yroot, zroot = pose_3d[5, 0], pose_3d[5, 1], pose_3d[5, 2]

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def draw_pose(points=None):
    fig = plt.figure(figsize=(12, 8), dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()
    minx = np.min(points[:, :, 0])
    maxx = np.max(points[:, :, 0])
    miny = np.min(points[:, :, 1])
    maxy = np.max(points[:, :, 1])
    minz = np.min(points[:, :, 2])
    maxz = np.max(points[:, :, 2])

    ax.set_xlim(left=minx - 1, right=maxx + 1)
    ax.set_ylim(miny - 1, maxy + 1)
    ax.set_zlim(minz - 1, maxz + 1)
    ax.margins(0)
    # 输入x、y、z轴范围之比，即可实现三轴等比例
    plt.gca().set_box_aspect((maxx-minx+2, maxy-miny+2, maxz-minz+2))

    i = 0
    while i < points.shape[0]:
        # plt.cla()  # 清除
        draw3Dpose(points[i], ax)
        # print(time.time() - t)
        plt.pause(0.001)
        i += 1
        # if i == points.shape[0]:
        #     i = 0
    plt.ioff()
    plt.show()
