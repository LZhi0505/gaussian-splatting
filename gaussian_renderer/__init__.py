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
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer   # 对应代码在 diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py中
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.general_utils import build_rotation
import torch.nn.functional as F
import matplotlib.pyplot as plt

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None,
           return_depth=False, return_normal=False, return_opacity=False, use_trained_exp=False):
    """
    将3D高斯投影到当前相机的2D图像平面上，生成渲染图像
        viewpoint_camera: 当前相机
        pc:     3D高斯模型
        pipe:   存储与渲染管线相关参数的args
        bg_color: 表征背景颜色的tensor，维度为(3,)，默认为黑色(0,0,0)，必须在GPU上
        scaling_modifier:
        override_color: 预先提供的用于覆盖的颜色，默认为None，则表示预先未提供颜色
    """

    # 创建一个与3D高斯模型大小相同的tensor，并初始化为0，记录屏幕空间中的点的位置，将用它来让 PyTorch 返回对于2D屏幕空间坐标的梯度的均值
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0

    try:
        # 尝试保留张量的梯度，确保在反向传播过程中 计算对于屏幕空间坐标的 梯度
        screenspace_points.retain_grad()
    except:
        pass

    # 计算视场角的tan值，用于设置光栅化配置
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5) # W = 2fx * tan(Fovx/2)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # 设置光栅器的配置
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,    # 背景颜色
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,   # W2C 的变换矩阵（视图矩阵）
        projmatrix=viewpoint_camera.full_proj_transform,    # W2NDC 的变换矩阵（投影矩阵）
        sh_degree=pc.active_sh_degree,  # 当前3D高斯模型的的球谐阶数
        campos=viewpoint_camera.camera_center,  # 当前相机中心在世界坐标系中的坐标
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    # 创建一个可微光栅器，将高斯分布投影到屏幕上
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz            # 所有高斯中心的 世界坐标
    means2D = screenspace_points    # 所有高斯中心投影到当前相机图像平面上的二维坐标
    opacity = pc.get_opacity        # 所有高斯的 不透明度（sigmoid激活后的）

    # 计算所有高斯的3D协方差矩阵 or 获取所有高斯的缩放因子和旋转四元数，在光栅化阶段计算（并行，速度快）
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        # 若希望在在python代码中计算3D协方差矩阵，则调用GaussianModel/build_covariance_from_scaling_rotation函数从从 缩放因子、旋转四元数 构建 所有高斯的 3D协方差矩阵
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # 默认
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # 计算所有高斯在当前相机观测下的RGB颜色值 or 获取所有高斯的球谐系数，在光栅化阶段转换为RGB颜色值（并行，速度快）
    shs = None
    colors_precomp = None
    if override_color is None:
        # 默认，未提供预先计算的颜色，则需后续计算颜色
        if pipe.convert_SHs_python:
            # 若希望在python代码中从球谐函数计算RGB颜色，则计算
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2) # 将所有高斯的球谐系数的维度调整为（N, 3, (max_sh_degree+1)**2）
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))  # 相机中心 到 每个高斯中心的方向向量
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)    # 归一化
            # 根据球谐系数、当前相机的观测方向，计算所有高斯的RGB颜色值
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0) # 将RGB颜色的范围限制在0到1之间
        else:
            # 默认
            shs = pc.get_features   # (N,16,3)
    else:
        # 提供了预先计算的颜色，则使用它
        colors_precomp = override_color

    # 调用光栅器将 在视野范围内的3D高斯投影到图像平面上，获取渲染图像 和 每个3D高斯在图像平面上的半径(radii)
    rendered_image, radii, depth_image = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # 返回一个字典，包含渲染的图像、屏幕空间坐标、可见性过滤器（根据半径判断是否可见）以及每个3D高斯投影到图像平面上的半径
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "depth": depth_image
    }

    if return_depth:
        #　提取相机的世界视图变换矩阵的第3列(深度方向)的前3个元素和最后一个元素。这些值将用于计算深度信息
        projvect1 = viewpoint_camera.world_view_transform[:, 2][:3].detach()
        projvect2 = viewpoint_camera.world_view_transform[:, 2][-1].detach()
        #　计算每个3D点的深度值：　means3D * projvect1.unsqueeze(0) 将 means3D (3D点坐标) 与 projvect1 (深度方向的前3个元素) 相乘,得到深度分量；
        #                   　深度分量求和,并加上 projvect2 (深度方向的最后一个元素),得到最终的深度值
        means3D_depth = (means3D * projvect1.unsqueeze(0)).sum(dim=-1, keepdim=True) + projvect2
        # 深度值复制成3个通道,与 colors_precomp 的尺寸匹配
        means3D_depth = means3D_depth.repeat(1, 3)
        render_depth, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=means3D_depth,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
        render_depth = render_depth.mean(dim=0) # 将批量维度上的深度值取平均,得到最终的深度图
        out.update({'render_depth': render_depth})

        # plt.figure(figsize=(8, 6))
        # plt.imshow(rendered_image.permute(1, 2, 0).detach().cpu().numpy(), cmap='viridis')
        # plt.colorbar()
        # plt.title('Rendered Image Map')
        # plt.show()

        # plt.figure(figsize=(8, 6))
        # plt.imshow(render_depth.detach().cpu().numpy(), cmap='viridis')
        # plt.colorbar()
        # plt.title('Rendered Depth Map')
        # plt.show()


    if return_normal:
        rotations_mat = build_rotation(rotations)
        scales = pc.get_scaling
        min_scales = torch.argmin(scales, dim=1)
        indices = torch.arange(min_scales.shape[0])
        normal = rotations_mat[indices, :, min_scales]

        # convert normal direction to the camera; calculate the normal in the camera coordinate
        view_dir = means3D - viewpoint_camera.camera_center
        normal = normal * ((((view_dir * normal).sum(dim=-1) < 0) * 1 - 0.5) * 2)[..., None]

        R_w2c = torch.tensor(viewpoint_camera.R.T).cuda().to(torch.float32)
        normal = (R_w2c @ normal.transpose(0, 1)).transpose(0, 1)

        render_normal, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=normal,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
        render_normal = F.normalize(render_normal, dim=0)
        out.update({'render_normal': render_normal})

        # plt.figure(figsize=(8, 6))
        # plt.imshow(render_normal.permute(1, 2, 0).detach().cpu().numpy(), cmap='viridis')
        # plt.colorbar()
        # plt.title('Rendered Normal Map')
        # plt.show()

    if return_opacity:
        density = torch.ones_like(means3D)

        render_opacity, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=density,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
        out.update({'render_opacity': render_opacity.mean(dim=0)})

    return out