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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R  # 相机到世界的 C2W
        self.T = T  # 世界到相机的 W2C
        self.FoVx = FoVx    # 水平方向上视场角
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        # 相机光心 到 视锥体最近平面、最远平面的距离，只有znear和zfar之间 且 在视锥内的高斯体才会被渲染
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans  # 相机中心的平移
        self.scale = scale  # 相机中心坐标的缩放

        # W 2 C 的变换矩阵：getWorld2View获取的就是是 W2C的变换矩阵，(后面的转置不是代表 C2W，而是为了后面的矩阵运算)，[4, 4]
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        # C 2 NDC 的变换矩阵：将相机坐标系中的3D点 转换到 NDC坐标系中，[4, 4]（NDC点投影到相机平面中的代码在cuda中）
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        # W 2 NDC 的变换矩阵：bmm只对输入矩阵的最后两个维度执行矩阵乘，即[1, 4, 4] @ [1, 4, 4] = [1, 4, 4] ==> [4, 4]
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3] # C2W的变换矩阵的平移向量 即为 相机光心在世界坐标系中的位置

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

