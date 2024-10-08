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
import sys
from datetime import datetime
import numpy as np
import random


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def PILtoTorch(pil_image, resolution):
    """
    调整图像分辨率，转为tensor，并归一化
    """
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0  # 归一化
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)  # 转换为 3 H W
    else:
        # 若为H W，则添加一个通道维度为 H W 1，再转换为 1 H W
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def get_expon_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000):
    """
    创建并返回一个学习率调整器 helper，并根据当前迭代次数 step动态调整学习率。step = 0返回 lr_init，step = max返回 lr_final，其余步数使用对数线性插值（指数衰减）
        lr_init:    初始学习率
        lr_final:   最终学习率
        lr_delay_steps: 学习率延迟步数，若>0，则学习率将被lr_delay_mult的平滑函数缩放，使得lr_init在优化开始时为lr_init * lr_delay_mult；当step>lr_delay_steps时，学习率将被缓慢恢复到正常学习率
        lr_delay_mult:  学习率延迟乘数，用于计算初始延迟学习率
        max_steps:  调整学习率所需的最大步数
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # 当前步数<0 或 学习率为0，直接返回0，表示不优化该参数
            return 0.0
        if lr_delay_steps > 0:
            # 设置了学习率延迟步数，则计算 延迟调整后的学习率，帮助在训练开始时使用较小的学习率,然后逐步恢复到正常水平,从而提高模型收敛的稳定性
            delay_rate = (
                    lr_delay_mult +
                    (1 - lr_delay_mult) * np.sin(0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))
            )
        else:
            delay_rate = 1.0

        # 根据步数计算学习率的对数线性插值，实现从初始学习率到最终学习率的平滑过渡
        t = np.clip(step / max_steps, 0, 1) # 当前步数占总步数的比例
        log_lerp = np.exp( np.log(lr_init) * (1 - t) + np.log(lr_final) * t )   # 对数线性插值（指数衰减）

        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    """
    从协方差矩阵中提取6个上半对角元素，节省内存
    [ _ _ _ ]
    [   _ _ ]
    [     _ ]
    """
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")  # N 6

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    """
    提取协方差矩阵的上半对角元素
        sym: 协方差矩阵
        return: 上半对角元素
    """
    return strip_lowerdiag(sym)


def build_rotation(r):
    """
    旋转四元数 -> 单位化 -> 3x3的旋转矩阵

    对于三维向量v，通过 四元数q = [r, x, y, z] (x, y, z为旋转轴的方向，2arccos(r)为要旋转的角度；q^-1 = q^* = [r, -x, -y, -z]) 得到的旋转结果为:
    v' = q v q^-1
       = R v
       = [ 1-2(y^2+z^2)  2(xy - rz)    2(xz + ry)   ]
         [ 2(xy + rz)    1-2(x^2+z^2)  2(yz - rx)   ] v
         [ 2(xz - ry)    2(yz + rx)    1-2(x^2+y^2) ]
    """
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    """
    构建3D高斯模型的 旋转-缩放矩阵
        s: 缩放因子, N 3
        r: 旋转四元素, N 4
        return: 旋转-缩放矩阵
    """
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")  # 初始化 缩放矩阵 为0，N 3 3
    R = build_rotation(r)  # 旋转四元数 -> 旋转矩阵，N 3 3

    # 构建缩放矩阵，其对角线元素对应为缩放因子的s1, s2, s3
    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L  # 高斯体的变化：旋转矩阵 乘 缩放矩阵
    return L


def safe_state(silent):
    old_f = sys.stdout

    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    # 若args.quiet 为 True，不写入任何文本到标准输出管道
    sys.stdout = F(silent)

    # 设置随机种子，使得结果可复现
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))  # torch 默认的 CUDA 设备为 cuda:0
