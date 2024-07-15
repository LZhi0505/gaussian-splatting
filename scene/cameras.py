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
from utils.general_utils import PILtoTorch
import cv2

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R  # 相机到世界的 C2W
        self.T = T  # 世界到相机的 W2C
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        resized_image_rgb = PILtoTorch(image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        self.alpha_mask = None
        if resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else:
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None and depth_params is not None and depth_params["scale"] > 0:
            invdepthmapScaled = invdepthmap * depth_params["scale"] + depth_params["offset"]
            invdepthmapScaled = cv2.resize(invdepthmapScaled, resolution)
            invdepthmapScaled[invdepthmapScaled < 0] = 0
            if invdepthmapScaled.ndim != 2:
                invdepthmapScaled = invdepthmapScaled[..., 0]
            self.invdepthmap = torch.from_numpy(invdepthmapScaled[None]).to(self.data_device)

            if self.alpha_mask is not None:
                self.depth_mask = self.alpha_mask.clone()
            else:
                self.depth_mask = torch.ones_like(self.invdepthmap > 0)

            if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                self.depth_mask *= 0
            else:
                self.depth_reliable = True

        # 距离相机平面znear和zfar之间且在视锥内的物体才会被渲染
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans  # 相机中心的平移
        self.scale = scale  # 相机中心坐标的缩放

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda() # getWorld2View获取的是 W2C的变换矩阵（需确认的是这的转置是代表 C2W，还是仅是为了后面的运算）
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()    # 生成了一个投影矩阵，用于将视图坐标投影到图像平面
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0) # 世界坐标系到图像坐标系的变换矩阵：使用 bmm（批量矩阵乘法）将世界到视图变换矩阵和投影矩阵相乘，生成完整的投影变换矩阵
        self.camera_center = self.world_view_transform.inverse()[3, :3] # 通过求逆变换矩阵获取 相机在世界坐标系中的位置（相机中心）

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

