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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation


class GaussianModel:
    def setup_functions(self):
        """
        定义和初始化处理高斯体模型参数的 激活函数
        """

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            """
            从 缩放因子、旋转四元数 构建 所有高斯的 3D协方差矩阵
            """
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)  # 从缩放因子、旋转四元数 计算RS矩阵（反映高斯体的变化），N 3 3
            actual_covariance = L @ L.transpose(1, 2)  # 计算实际的 协方差矩阵 R S S^T R^T
            symm = strip_symmetric(actual_covariance)  # 提取上半对角元素
            return symm

        # 初始化一些激活函数
        self.scaling_activation = torch.exp             # 缩放因子的激活函数，exp函数，确保缩放因子 非负
        self.scaling_inverse_activation = torch.log     # 缩放因子的逆激活函数，用于梯度回传，log函数

        self.covariance_activation = build_covariance_from_scaling_rotation  # 协方差矩阵的激活函数（实际未使用激活函数，直接构建）

        self.opacity_activation = torch.sigmoid             # 不透明度的激活函数，sigmoid函数，确保不透明度在0到1之间
        self.inverse_opacity_activation = inverse_sigmoid   # 不透明度的逆激活函数

        self.rotation_activation = torch.nn.functional.normalize  # 旋转四元数的激活函数，归一化函数（取模）

    def __init__(self, sh_degree: int):
        """
        初始化3D高斯模型的参数 和 激活函数
            sh_degree: 设定的 球谐函数的最大阶数，默认为3，用于控制颜色表示的复杂度
        """
        self.active_sh_degree = 0       # 当前球谐函数的阶数，初始为0
        self.max_sh_degree = sh_degree  # 允许的最大球谐阶数j

        self._xyz = torch.empty(0)      # 各3D高斯的 中心位置

        self._features_dc = torch.empty(0)      # 球谐函数的 直流分量，第一个元素，用于表示基础颜色
        self._features_rest = torch.empty(0)    # 球谐函数的 高阶分量，用于表示颜色的细节和变化

        self._scaling = torch.empty(0)      # 各3D高斯的 缩放因子，控制高斯的形状
        self._rotation = torch.empty(0)     # 各3D高斯的 旋转四元数
        self._opacity = torch.empty(0)      # 各3D高斯的 不透明度（sigmoid前的），控制可见性
        self.max_radii2D = torch.empty(0)   # 各3D高斯 投影到2D平面上的最大半径

        self.xyz_gradient_accum = torch.empty(0)    # 3D高斯中心位置 梯度的累积值，当它太大的时候要对高斯体进行分裂，太小代表under要复制
        self.denom = torch.empty(0)                 # 与累积梯度配合使用，表示统计了多少次累积梯度，用于计算每个高斯体的平均梯度时需除以它（denom = denominator，分母）

        self.optimizer = None  # 优化器，用于调整上述参数以改进模型（论文中采用Adam，见附录B Algorithm 1的伪代码）

        self.percent_dense = 0      # 百分比密度，控制3D高斯的密度，默认为0.01
        self.spatial_lr_scale = 0   # 各3D高斯的 位置学习率的变化因子，位置的学习率乘以它，以抵消在不同尺度下应用同一个学习率带来的问题

        # 初始化高斯体模型各参数的 激活函数
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        # 从chkpnt中加载相关训练参数
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)  # 设置相关参数、配置优化器、创建位置学习率调整器
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):  # 获取的是激活后的 缩放因子
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):  # 获取的是激活后的 旋转四元数
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_opacity(self):  # 获取的是激活后的 不透明度
        return self.opacity_activation(self._opacity)

    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        # 当前球谐函数的阶数 < 规定的最大阶数，则 阶数+1
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        """
        从输入点云创建3D高斯，初始化各3D高斯的参数
            pcd: 输入点云，包含点的位置和颜色
            spatial_lr_scale: 位置学习率的 变化因子
        """
        # 根据scene.Scene.__init__ 以及 scene.dataset_readers.SceneInfo.nerf_normalization，即scene.dataset_readers.getNerfppNorm的代码，
        # 这个值似乎是训练相机中离它们的坐标平均值（即中心）最远距离的1.1倍，根据命名推断应该与学习率有关，防止固定的学习率适配不同尺度的场景时出现问题。
        self.spatial_lr_scale = spatial_lr_scale

        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()  # 点云的3D位置从array转换为tensor，并放到cuda上，(N, 3)

        # 点云的颜色从RGB转为球谐函数直流分量的系数 (N, 3)，因为只有点的颜色
        fused_color = RGB2SH(
            torch.tensor(np.asarray(pcd.colors)).float().cuda()
        )
        # 初始化球谐函数的系数，RGB每个通道有(max_sh_degree + 1)**2 个系数
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()  # (N, 3, 16)
        features[:, :3, 0] = fused_color  # 将RGB转换后的球谐系数C0项的系数(直流分量)存入每个3D高斯的直流分量球谐系数中
        features[:, :3, 1:] = 0.0         # 其余高阶分量系数先初始化为0，后续在优化阶段再赋值

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # 计算点云中 每个点到其最近的k个点平均距离的平方 (N, )，用于确定高斯的缩放因子scale，且不能<1e-7
        dist2 = torch.clamp_min(
            distCUDA2( torch.from_numpy(np.asarray(pcd.points)).float().cuda() ), # 由submodules/simple-knn/simple_knn.cu的 SimpleKNN::knn()函数实现，KNN意思是K-Nearest Neighbor，即求每一点距其最近K个点平均距离的平方
            0.0000001
        )
        # 初始化各3D高斯的 缩放因子，repeat(1, 3) 表明三个方向的初值都先设为平均距离（因dist2其实是距离的平方，所以这里要开根号；因为取值时都是经激活后的值，而scale的激活函数是exp，则这里先取对数存值
        scales = torch.log( torch.sqrt(dist2) )[..., None].repeat(1, 3)  # (N, 3)

        # 初始化各3D高斯的 旋转因子 为单位四元数，且无旋转
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")  # (N, 4)
        rots[:, 0] = 1  # 四元数[r, x, y, z]的实部为1，则旋转角度为2arccos(1)，表示无旋转

        # 初始化各3D高斯的 不透明度为0.1，(N, 1)（不透明度的激活函数是sigmoid，所以先取逆对数存值，inverse_sigmoid(0.1) = -2.197）
        opacities = self.inverse_opacity_activation(
            0.1 * torch.ones( (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda" )
        )

        # 将以上需计算的参数设置为模型的可训练参数
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))    # 各3D高斯的中心位置，(N, 3)
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))     # 球谐函数直流分量的系数，(N, 1, 3)
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))    # 球谐函数高阶分量的系数，(N, (最大球谐阶数 + 1)² - 1, 3)
        self._scaling = nn.Parameter(scales.requires_grad_(True))       # 缩放因子，(N, 3)
        self._rotation = nn.Parameter(rots.requires_grad_(True))        # 单位旋转四元数，(N, 4)
        self._opacity = nn.Parameter(opacities.requires_grad_(True))    # 不透明度，(N, 1)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")  # 投影到所有2D图像平面上的最大半径，初始化为0，(N, )
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        """
        初始设置要训练的3D高斯参数，包括初始化用于累积梯度的变量，配置优化器，以及创建位置学习率调整器
            training_args: 包含优化相关参数的对象
        """
        # 在训练过程中，用于控制3D高斯的密度，在`densify_and_clone`中被使用
        self.percent_dense = training_args.percent_dense

        # 初始化 累积3D高斯中心点位置梯度的张量，用于之后判断是否需要对3D高斯进行克隆或切分
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")  # 坐标的累积梯度

        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")  # 梯度累积的次数，用于计算每个高斯体的平均梯度时需除以它

        # 配置各参数的优化器，包括指定参数、学习率和参数名称
        l = [
            {"params": [self._xyz], "lr": training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},    # 各3D高斯的位置学习率的 变化因子，位置的学习率乘以它，以抵消在不同尺度下应用同一个学习率带来的问题
            {"params": [self._features_dc], "lr": training_args.feature_lr, "name": "f_dc"},
            {"params": [self._features_rest], "lr": training_args.feature_lr / 20.0, "name": "f_rest"},
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": training_args.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
        ]

        # 创建优化器，这里使用Adam优化器，初始学习率为0
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        if self.pretrained_exposures is None:
            self.exposure_optimizer = torch.optim.Adam([self._exposure])

        # 创建一个位置学习率调整器，用于调整各3D高斯位置的学习率
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult, #
            max_steps=training_args.position_lr_max_steps,      #
        )

        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        # 每次迭代都更新3D高斯的位置学习率
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)  # 调整学习率
                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        """
        构建ply文件的key列表
        模型被保存为一个.ply文件，使用PlyData.read()读取，其第一个属性，即vertex的信息为：'x', 'y', 'z', 'nx', 'ny', 'nz', 3*1个'f_dc_x', 3*15个'f_rest_xx', 'opacity', 3个'scale_x', 4个'rot_x'
        """
        l = ["x", "y", "z", "nx", "ny", "nz"]  # 添加位置、法向量（法向量这里实际没有用到，全为0）
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):        # 添加球谐函数直流分量的系数，3 * 1
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):    # 添加球谐函数高阶分量的系数，3 * 15
            l.append("f_rest_{}".format(i))
        l.append("opacity") # 添加不透明度，1
        for i in range(self._scaling.shape[1]):     # 添加缩放，3
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):    # 添加旋转四元数，4
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        """
        保存3D高斯模型为ply文件
        """
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)    # nx, ny, nz在保存时全为0，实际未用到
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()   # N 1 3 => N 3 1 => N 3
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()   # N 15 3 => N 3 15 => N 45
        opacities = self._opacity.detach().cpu().numpy()    # N 1
        scale = self._scaling.detach().cpu().numpy()        # N 3
        rotation = self._rotation.detach().cpu().numpy()    # N 4

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]   # 遍历参数名称，创建一个元组，key为参数名，value为字符串"f4"，表示float32数据类型
        elements = np.empty(xyz.shape[0], dtype=dtype_full) # 创建一个空的 结构化数组 存储所有参数，(N,)

        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)  # 所有要保存的参数数据先concatenate，(N, x)
        elements[:] = list(map(tuple, attributes))  # map：将每一行即每个3DGS的各参数数据转换为一个元组，list的每个元素为元组形式存储的一个3DGS的各参数数据
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_opacity(self):
        """
        重置不透明度，让所有3D高斯的不透明度都 < 0.01
        """
        # 取当前各3D高斯的不透明度 和 0.01的最小值（get_opacity返回的是经sigmod激活后的不透明度，后需经过反函数）
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))  # N 1
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")  # 更新优化器中的不透明度，返回的是一个字典中，key："opacity"，value：可计算梯度的、参数化的新不透明度
        self._opacity = optimizable_tensors["opacity"]  # 存储新不透明度值

    def load_ply(self, path, use_train_test_exp = False):
        """
        读取ply文件，并把数据转换成torch.nn.Parameter等待优化
        """
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None
        # N 3
        xyz = np.stack(
            (np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"])),
            axis=1
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis] # N 1

        features_dc = np.zeros((xyz.shape[0], 3, 1))  # N 3 1
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        # 读取球谐函数高阶分量的系数
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]  # 先获取参数名，3*15个
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3  # (3 * 16) - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))  # N 3*15
        # 将高阶份量系数存储到features_extra中
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P, F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))  # N 3 15

        # 读取缩放因子
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))  # N 3
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # 读取旋转四元数
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))     # N 3 1 => N 1 3
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))    # N 3 15 => N 15 3
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree  # 设置当前球谐函数的阶数

    def replace_tensor_to_optimizer(self, tensor, name):
        """
        将优化器保存的名为`name`的参数的值 强行替换为 传入的`tensor`，同时重置Adam优化器对应参数的状态变量：动量、平方动量（确保该参数新的值从一个干净的状态开始，不受之前优化路径的影响，确保优化过程的稳定性和收敛速度）
        """
        optimizable_tensors = {}  # 存储新的优化参数及对应数据
        # 遍历优化器的所有参数组
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                # 当前参数组的名称 == 要修改的参数名称
                stored_state = self.optimizer.state.get(group["params"][0], None)   # 暂存当前要修改的参数组的 参数数据（group["params"][0]）的状态
                stored_state["exp_avg"] = torch.zeros_like(tensor)      # 将动量清零
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)   # 将平方动量清零

                del self.optimizer.state[group["params"][0]]  # 删除优化器中当前要修改的参数组 的旧数据的状态

                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))  # 旧数据替换为 输入的新数据，并设置为可计算梯度
                self.optimizer.state[group["params"][0]] = stored_state  # 将动量清零的旧数据的状态 重新分配给 优化器中的新数据

                optimizable_tensors[group["name"]] = group["params"][0]  # 将新的参数数据存储在optimizable_tensors字典中，key：参数名，value：可计算梯度的新数据
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        """
        保留`mask`为True对应位置的3D高斯，清除 优化器中的参数组中要被去除的3D高斯 对应的参数数据 和 状态中的动量
            mask：维度为 (N,)，为True表示要保留该3D高斯
        """
        optimizable_tensors = {}
        # 遍历优化器的所有参数组，替换数据为 mask为True对应3D高斯的数据
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)   # 暂存当前参数组的 参数数据（group["params"][0]）的状态
            if stored_state is not None:
                # 已有状态，则还需修改状态中的动量、平方动量
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]    # 删除旧数据的状态

                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))  # 旧数据替换为 只保留mask为True对应3D高斯的数据，并设置为可计算梯度
                self.optimizer.state[group["params"][0]] = stored_state # 替换为新状态

                optimizable_tensors[group["name"]] = group["params"][0] # 存储
            else:
                # 还没有状态，则只替换数据
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        """
        删除`mask`中为True对应位置的3D高斯，并移除它们的所有属性
            mask：维度为 (N,)，为True表示要去除该3D高斯
        """
        valid_points_mask = ~mask   # 要保留的3D高斯mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        # 重置高斯模型中各参数对应的张量
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        """
        将包含新3D高斯参数的张量字典 添加到优化器中
            return：新的参数张量
        """
        optimizable_tensors = {}
        # 遍历优化器的参数组
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1    # 确保每个参数组中仅包含一个参数
            extension_tensor = tensors_dict[group["name"]]  # 获取当前参数组对应的新张量
            stored_state = self.optimizer.state.get(group["params"][0], None)   # 获取当前参数组中第一个参数的优化器状态（如动量和平方动量）
            if stored_state is not None:
                # 当前参数有优化器状态，则更新状态：将动量、平方动量扩展，新部分初始化为0
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                del self.optimizer.state[group["params"][0]]    # 删除旧的优化器状态

                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))    # 将当前参数与新张量cat，创建新的参数张量
                self.optimizer.state[group["params"][0]] = stored_state     # 赋予新的状态

                optimizable_tensors[group["name"]] = group["params"][0]     # 暂存新的参数张量
            else:
                # 当前参数无优化器状态，则直接将当前参数与新张量cat，创建新的参数张量
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        """
        将新增3D高斯的参数张量添加到优化器中，更新高斯模型的各参数张量，重置梯度累加值、累加次数、投影到2D平面的最大半径为0
        """
        # 1. 将新增3D高斯的参数张量添加到优化器中，并返回新的参数张量
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        # 2. 更新高斯模型的各参数张量
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # 3. 重置 梯度累加值、累加次数、投影到2D平面的最大半径 为0
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")    # N 1
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")                 # N 1
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")              # N

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """
        通过`分裂`增稠（条件1 && 条件2），新3D高斯的位置是以原先的大高斯作为概率密度函数进行采样的，新3D高斯的缩放因子被除以φ=1.6，以减小尺寸
        条件1：累加梯度 >= 阈值
        条件2：最大缩放因子 > （控制密度的百分比，0.01）*（所有train相机包围圈的半径 * 1.1）
            grads: 训练到目前 所有3D高斯投影在2D图像平面各像素上累加的 梯度 的L2范数 平均每次累加的值，(N,)
            grad_threshold： 梯度阈值，默认为0.0002
            scene_extent：   所有train相机包围圈的半径 * 1.1
            N：              代表1个3D高斯分裂为N个，默认为2
        """
        n_init_points = self.get_xyz.shape[0]   # 当前3D高斯个数
        # Extract points that satisfy the gradient condition
        # 初始化 扩展后的3D高斯梯度为 0张量，并用实际当前梯度填充前面的部分
        padded_grad = torch.zeros((n_init_points), device="cuda")   # (N_3DGS,)
        padded_grad[: grads.shape[0]] = grads.squeeze()

        # 2. 标记出满足（条件1 && 条件2）的大3D高斯 分裂成 两个小3D高斯
        # 条件1：累加梯度 >= 阈值
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        # 条件2：最大缩放因子 > （控制密度的百分比，0.01）*（所有train相机包围圈的半径 * 1.1）
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)

        # 3. 分裂
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)     # 标准差：被分裂的大3D高斯的缩放因子 复制N-1次，[N_select3DGS, 3] ==> [N_selected3DGS * 2, 3]
        means = torch.zeros((stds.size(0), 3), device="cuda")       # 均值：初始化为0，[N_selected3DGS * 2, 3]
        samples = torch.normal(mean=means, std=stds)                # 生成标准正态分布的随机样本，均值为0，标准差为stds，[N_selected3DGS * 2, 3]
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)    # 被分裂的大3D高斯的旋转四元数转为旋转矩阵，并复制N-1次，[N_selected3DGS, 3, 3] ==> [N_selected3DGS * 2, 3, 3]

        # 新的两个小3D高斯的位置，以原先大高斯作为概率密度函数进行采样的：大3D高斯的旋转矩阵 @ 随机样本`sample`，再加上大3D高斯的位置，[N_selected3DGS * 2, 3, 3] @ [N_selected3DGS * 2, 3, 1] = [N_selected3DGS * 2, 3, 1] ==> [N_selected3DGS * 2, 3]（bmm: 批量矩阵乘法，对输入矩阵的最后两个维度执行矩阵乘法，其他维度不变）
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        # 新的两个小3D高斯的缩放因子，以原先大高斯的缩放因子除以 φ = 0.8 * N = 1.6得到
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)) # [N_selected3DGS * 2, 3]
        # 剩下的参数直接复制
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        # 将分裂出的两个新3D高斯的参数张量添加到优化器中，更新高斯模型的各参数张量，重置梯度累加值、累加次数、投影到2D平面的最大半径为0
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        # 创建一个剪枝mask，清除对应位置为True的3D高斯，即清除被分裂的原大3D高斯，不清除新分裂的小3D高斯
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """
        通过`克隆`增稠（条件1 && 条件2）
        条件1：累加梯度的L2范数 >= 阈值
        条件2：最大缩放因子 <= （控制密度的百分比，0.01）*（所有train相机包围圈的半径 * 1.1）
            grads: 训练到目前 所有3D高斯投影在2D图像平面各像素上累加的 梯度 的L2范数 平均每次累加的值，(N,)
            grad_threshold: 梯度阈值，默认为0.0002
            scene_extent:   所有train相机包围圈的半径 * 1.1
        """
        # 1. 标记出满足（条件1 && 条件2）的3D高斯进行克隆
        # 条件1：累加梯度的L2范数 >= 阈值
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # 条件2：最大缩放因子 <= （控制密度的百分比，0.01）*（所有train相机包围圈的半径 * 1.1）
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)

        # 2. 克隆
        # 2.1 复制被克隆的3D高斯的参数
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        # 2.2 将新增3D高斯的参数张量添加到优化器中；更新高斯模型的各参数张量；重置梯度累加值、累加次数、投影到2D平面的最大半径为0
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        """
        增稠和剪枝
            max_grad：       最大梯度阈值，即arguments中的densify_grad_threshold，默认为0.0002，决定是否应基于2D位置梯度对3D高斯进行增稠的阈值
            min_opacity：    最小不透明度阈值，默认为0.005
            extent：         所有train相机包围圈的半径 * 1.1
            max_screen_size：    <= 3000代为None；[3100, 14900]代为20
        """
        grads = self.xyz_gradient_accum / self.denom  # 计算从开始训练到当前迭代次数下 所有3D高斯投影在2D图像平面各像素上累加的 梯度 的L2范数 平均每次累加的值，(N,)
        grads[grads.isnan()] = 0.0  # NaN值设为0，保持数值稳定性

        # 1. 增稠
        self.densify_and_clone(grads, max_grad, extent)  # 对under reconstruction的区域 克隆 增稠（累加梯度的L2范数 >= 阈值 && 最大缩放因子 <= percent_dense(0.01)*所有train相机包围圈的半径*1.1）
        self.densify_and_split(grads, max_grad, extent)  # 对over reconstruction的区域 分裂 增稠（累加梯度 >= 阈值 && 最大缩放因子 > percent_dense*所有train相机包围圈的半径*1.1）

        # 2. 剪枝（不透明度 < 最低阈值 || 3D高斯太大 || 3D高斯针状）
        # 条件1：不透明度 < 最低阈值0.005
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            # 条件2：[3100, 14900]代，各3D高斯投影到所有2D图像平面上的最大半径 > 20个像素（在视图空间中太大要被去除）
            big_points_vs = self.max_radii2D > max_screen_size
            # 条件3：[3100, 14900]代，各3D高斯的最大缩放因子 > 0.1*所有train相机包围圈的半径*1.1（在世界空间中太大或针状要被去除）
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            # 逻辑或
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        # 清除 优化器中的参数组中满足（条件1 || 条件2 || 条件3）的3D高斯，包括对应的参数数据 及 状态中的动量
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()    # 清理GPU缓存，释放一些内存

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """
        累加所有3D高斯投影在2D图像平面各像素上 梯度 的L2范数，与累加次数
            viewspace_point_tensor: 各3D高斯投影到当前相机图像平面上的2D坐标
            update_filter: 各3D高斯投影到当前相机图像平面上半径>0的mask
        """
        # 选择投影到当前相机图像平面上半径>0的3D高斯 投影在2D平面坐标处的 x、y方向上的梯度，并计算这些梯度的L2范数；并在训练过程中一直累加
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)  # (N,)
        # 同时记录 梯度累加的次数
        self.denom[update_filter] += 1  # (N,)
