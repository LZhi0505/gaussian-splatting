#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    """
    计算W2C的变换矩阵
        R:  W2C的 旋转矩阵
        t:  W2C的 平移向量
    """
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()  # W2C的 R
    Rt[:3, 3] = t   # W2C的 t
    Rt[3, 3] = 1.0
    # C2W的变换矩阵
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3] # C2W的 t，即相机在世界坐标系中的坐标位置
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center # 加上平移和缩放后的 C2W的变换矩阵

    Rt = np.linalg.inv(C2W) # W2C的变换矩阵
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    """
    计算变换矩阵P，将 相机坐标系中的3D点 转换到 NDC坐标系中（可能是因为colmap允许每个相机拥有不同的内参，不同的视场角，所以这里将不同的相机统一到NDC坐标系中）
        znear，zfar: 相机光心 到 视锥体最近平面、最远平面的距离
        fovX，fovY:  水平、垂直方向上视场角
    """
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear   # 近平面顶部 到 相机光轴的 高度
    bottom = -top
    right = tanHalfFovX * znear # 近平面右部 到 相机光轴的 宽度
    left = -right

    P = torch.zeros(4, 4)   # 相机坐标系中的3D点 转换到 NDC坐标系中的 变换矩阵，P * [x, y, z, 1] ~= [x', y', z', 1]

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    '''
        focal: fx 或 fy
        pixels: 宽度或高度,单位为像素
    '''
    return 2 * math.atan(pixels / (2 * focal))