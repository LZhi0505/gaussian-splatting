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

import os
import numpy as np
import torch
from random import randint
from PIL import Image
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

# 尝试导入 PyTorch 提供的 TensorBoard 记录器 SummaryWriter 类
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    """
        dataset: 存储与3D高斯模型相关参数的args
        opt:    存储与优化相关参数的args
        pipe:   存储与渲染管线相关参数的args
        testing_iterations, saving_iterations, checkpoint_iterations：分别为测试3DGS的PSNR、保存3DGS、保存训练参数的迭代次数
        checkpoint: 要加载的训练参数所在的文件夹路径
        debug_from: 从哪一个迭代开始debug
    """
    first_iter = 0
    # 创建保存3D高斯模型的文件夹，并保存模型的相关参数设置到cfg_args文件；尝试创建tensorboard_writer，记录训练过程
    tb_writer = prepare_output_and_logger(dataset)

    # 创建3D高斯模型对象，初始化相关参数 和 激活函数
    gaussians = GaussianModel(dataset.sh_degree)
    # 创建3D场景对象，初始化相关参数，加载场景info，创建train和test相机，调整gt图像的分辨率，从稀疏点云初始化3D高斯或直接加载指定的3D高斯
    scene = Scene(dataset, gaussians)
    # 初始设置要训练的3D高斯参数，包括初始化用于累积梯度的变量，配置优化器，以及创建位置学习率调整器
    gaussians.training_setup(opt)

    if checkpoint:
        # 如果设置了checkpoint，则从checkpoint加载训练参数并恢复训练进度
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # 设置背景颜色，默认为黑色，对应0, 0, 0；白色则为1, 1, 1
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 创建CUDA事件用于计时
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0

    # 使用tqdm库创建进度条，追踪训练进度
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    first_iter += 1
    # 开始迭代
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()  # 记录当前迭代的开始时间

        gaussians.update_learning_rate(iteration)  # 根据当前迭代次数更新位置学习率

        # 每迭代1000次，提升球谐函数的阶数直至设置的最大阶数3，以提升3G高斯模型的复杂度、渲染质量
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 先获取所有train相机到viewpoint_stack，再从中随机选择一个
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if (iteration - 1) == debug_from:
            # 如果到达要debug的迭代次数，则启用debu模式
            pipe.debug = True

        # 参数设置了随机背景颜色，则随机生成；否则为 白色 或 黑色（默认）
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # 使用可微光栅化器获取当前train相机视角下的渲染数据
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, return_depth=True, return_normal=True)
        # 分别为：渲染图像，所有高斯中心 2D投影像素坐标（实际只用于接收梯度），所有高斯是否被当前相机可见的mask，所有高斯投影到当前相机图像平面上的半径
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        gt_image = viewpoint_cam.original_image.cuda()

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        loss.backward() # 触发反向传播，计算loss对所有可求导参数的梯度，并将这些梯度存储到它们的.grad属性中

        iter_end.record()  # 记录当前迭代的结束时间

        with torch.no_grad():
            # 记录loss的指数移动平均值，并定期更新进度条
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                # 每迭代10次，更新一次进度条
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                # 到达设定的训练次数，关闭进度条
                progress_bar.close()

            # 定期测试模型训练情况
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))

            if (iteration in saving_iterations):
                # 到达要保存ply模型的迭代次数，则保存
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                # 迭代次数 < 增稠停止的迭代次数(默认为15000)，则对3D高斯模型进行增稠和剪枝（根据所有高斯投影图像平面的最大半径以进行修剪）

                # 使用各高斯投影到当前相机的最大半径 更新 其投影到所有相机图像平面的最大半径的最高值（max_radii2D：记录的 各高斯投影到所有相机图像平面上的最大半径的最高值；radii：各高斯投影到当前相机图像平面的最大半径；visibility_filter：各3D高斯投影到当前相机图像平面上半径>0的mask）
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

                # 累加 loss对各高斯中心2D投影位置梯度 的L2范数 与 次数
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # 迭代次数 > 增稠开始的迭代次数(默认为500)，即[600, 14900]每迭代100次进行一次增稠和剪枝

                    # 获取各高斯投影在所有相机图像平面的最大半径的 像素阈值
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None  # <= 3000代为None；[3100, 14900]代为20

                    # 增稠 和 剪枝
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                # 每迭代3000次 或 白背景且第一次增稠前(500代)，则重置所有3D高斯的不透明度（ < 0.01），防止相机附近出现伪影
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # 使用优化器根据梯度更新各参数
            if iteration < opt.iterations:
                gaussians.optimizer.step()  # 根据各参数的梯度 调整参数
                # 在每次迭代中，需从零开始计算新的梯度，否则梯度会累积，导致错误的更新
                gaussians.optimizer.zero_grad(set_to_none=True) # 将所有参数的梯度设置为 None。相比于置零，因不需要创建新的零张量，可以减少内存占用

            # 到达要保存训练数据的迭代次数时，保存相应迭代次数的 高斯模型、优化器的参数和状态、高斯中心位置的学习率
            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):
    if not args.model_path:
        # 没有预设保存3D高斯模型的输出路径，则随机生成一个文件名，存储输出结果
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # 创建输出文件夹
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    # 保存当前3DGS模型的配置设置
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # 创建Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)  # 创建一个 SummaryWriter 对象,向 TensorBoard 记录训练过程中的各种指标
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs):
    """
        tb_writer:
        iteration:  当前迭代次数
        Ll1:        当前迭代的 L1 loss
        loss:       当前迭代的总 loss
        l1_loss:    L1 loss计算函数
        elapsed:    当前迭代中的 训练时间
        testing_iterations: 要测试PSNR的迭代次数列表
        scene:      3D场景
        renderFunc: 渲染函数
        renderArgs: (与渲染管线相关的参数，背景颜色默认黑色)
    """
    if tb_writer:
        # 将 L1 loss、总体 loss 和迭代时间写入 TensorBoard。
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    if iteration in testing_iterations:
        # 到达指定的测试迭代次数，渲染并计算 L1 loss 和 PSNR
        torch.cuda.empty_cache()
        # 渲染所有的test相机，从train相机中选择5个相机：第5, 10, 15, 20, 25个
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {"name": "train", "cameras": [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]},
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                # 渲染train/test相机
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    # 遍历每个相机视角，获取当前视角下的渲染图像(限制到0,1) 与 gt图像
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        # 在TensorBoard中记录渲染结果和真实图像
                        tb_writer.add_images(config["name"] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration
                            )
                    # 计算 L1 loss 和 PSNR
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                # 平均
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config["name"], l1_test, psnr_test))

                if tb_writer:
                    # 在 TensorBoard 中记录评估结果
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration)

        if tb_writer:
            # 在 TensorBoard 中记录场景的不透明度直方图 和 总点数
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")

    # 创建 模型、优化、渲染 相关参数的对象
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6007)
    parser.add_argument("--debug_from", type=int, default=-1)  # 指定从哪一迭代（>= 0）开始debug
    parser.add_argument("--detect_anomaly", action="store_true", default=False)  # 是否检测梯度异常，其中action='store_true'表示如果命令行中包含了这个参数,它的值将被设为True
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default=None)  # 要加载的训练参数所在的文件夹路径

    # 将命令行参数覆盖parser内的参数，并存储到args
    args = parser.parse_args(sys.argv[1:])

    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # 初始化系统的随机状态,以确保实验结果可复现 (RNG)
    safe_state(args.quiet)

    # 启动 GUI 服务器, 监听指定的 IP 地址和端口，可以使用SIBR查看器观察训练进度和调试问题
    network_gui.init(args.ip, args.port)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)  # 设置pytorch是否检测梯度异常

    # lp.extract(args)：args中参数 覆盖 模型、优化、渲染 的参数，并形成新的args
    lp_args = lp.extract(args)
    op_args = op.extract(args)
    pp_args = pp.extract(args)

    training(lp_args, op_args, pp_args, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
