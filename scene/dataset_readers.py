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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict  # 逆深度对齐参数
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool   # 是否是测试相机

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool

def getNerfppNorm(cam_info):
    """
    计算所有相机的 中心点坐标、所有相机到该点的最大距离
    """
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)    # 计算所有train相机的 中心点
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)  # 计算每个相机到中心点的 距离
        diagonal = np.max(dist) # 距离的最大值
        return center.flatten(), diagonal

    cam_centers = []
    # 遍历所有相机信息，获取中心坐标
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)  # 计算W2C的变换矩阵
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])    # 获取相机的中心坐标（Twc的平移向量）

    center, diagonal = get_center_and_diag(cam_centers) # 计算所有相机的 中心点、所有相机到该点的最大半径
    radius = diagonal * 1.1 # 最终的半径 = 最大半径 * 1.1

    translate = -center # 所有相机的 中心点 的tcw

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list):
    '''
        cam_extrinsics: 存储所有相机的 外参的字典，每个元素包括：id(图片ID)、qvec(W2C的旋转四元数)、tvec(W2C的平移向量)、camera_id(相机ID)、name(图像名)、xys(所有特征点的像素坐标)、point3D_ids(所有特征点对应3D点的ID，特征点没有生成3D点的ID则为-1)
        cam_intrinsics: 存储所有相机的 内参的字典，每个元素包括：id(相机ID)、model(相机模型ID)、width、height、params(内参数组)
        depths_params:  逆深度图对齐的 有效参数
        images_folder:  保存RGB图的文件夹路径
        depths_folder:  保存逆深度图的文件夹路径
        test_cam_names_list: 测试相机 图片名列表
    '''
    # 初始化存储相机信息类CameraInfo对象的列表
    cam_infos = []

    # 遍历所有相机的外参
    for idx, key in enumerate(cam_extrinsics):
        # 动态显示读取相机信息的进度
        sys.stdout.write('\r')  # 光标回到当前行的最前面
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()  # 立即将缓冲区中的内容输出到控制台

        # 获取当前相机的外参和内参
        extr = cam_extrinsics[key]  # 当前相机的外参类Imgae对象
        intr = cam_intrinsics[extr.camera_id]   # 根据外参中的camera_id找到对应的内参类对象
        height = intr.height    # 图片高度
        width = intr.width      # 图片宽度

        uid = intr.id   # 相机的唯一标识符

        R = np.transpose(qvec2rotmat(extr.qvec))    # W2C的四元数转为 R，transpose后 ==> C2W的R
        T = np.array(extr.tvec)     # W2C的T

        # 根据相机内参模型计算 视场角（FoV）
        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
            # 如果是简单针孔模型，只有一个焦距参数
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)    # 计算垂直方向的视场角: 2 * arctan(H / 2fy))
            FovX = focal2fov(focal_length_x, width)     # 计算水平方向的视场角: 2 * arctan(W / 2fx)
        elif intr.model=="PINHOLE":
            # 如果是针孔模型，有两个焦距参数
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)    # 使用fy计算垂直视场角
            FovX = focal2fov(focal_length_x, width)     # 使用fx计算水平视场角
        else:
            # 如果不是以上两种模型，抛出错误
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1    # 图像名中 格式名称+'.'的个数
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]] # 当前相机的 逆深度对齐参数
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        if not os.path.exists(image_path):
            continue

        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        # 创建相机信息类CameraInfo对象，并添加到列表cam_infos中
        # 这里添加的 width、heigth 是COLMAP时的内参中的
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)
    # 在读取完所有相机信息后换行
    sys.stdout.write('\n')
    print("valid Colmap camera size: {}".format(len(cam_infos)))

    return cam_infos

def fetchPly(path):
    # 读取.ply文件
    plydata = PlyData.read(path)
    # 其第一个属性，即vertex的信息为：x', 'y', 'z', 'nx', 'ny', 'nz', 3个'f_dc_x', 45个'f_rest_xx', 'opacity', 3个'scale_x', 4个'rot_x'
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    '''
    加载COLMAP的结果中的二进制相机外参文件imags.bin 和 内参文件cameras.bin
        path:   source_path
        images: "images"
        depths: 存储深度图文件夹的相对路径，参照"images"，默认为 ""
        eval:   是否为eval模式
        train_test_exp: 是否考虑 光照补偿
        llffhold: 采样频次，默认为8，即每8张中取第1张
    '''

    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file) # 存储所有相机 外参信息 的字典，每个元素包括：id(图片ID)、qvec(W2C的旋转四元数)、tvec(W2C的平移向量)、camera_id(相机ID)、name(图像名)、xys(所有特征点的像素坐标)、point3D_ids(所有特征点对应3D点的ID，特征点没有生成3D点的ID则为-1)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file) # 存储所有相机 内参信息 的字典，每个元素包括：id(相机ID)、model(相机模型ID)、width、height、params(内参数组)
    except:
        # 如果bin文件读取失败，尝试读取txt格式的相机外参和内参文件
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    # 读取逆深度图对齐的 有效参数
    # 以字典形式存储的 前期计算的 深度估计逆深度 对齐到 COLMAP尺度的 scale和offset的文件。key: 图像名，value: 一个字典，包含scale、offset
    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        # 传入了深度图路径，则尝试读取尺度对齐参数
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            # 获取所有图像的 scale
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            # scale的 中值
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0

            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    # 根据是否eval，将相机分为训练集和测试集
    if eval:
        if "360" in path:
            # MipNeRF360 数据集，则llffhold = 8
            llffhold = 8
        if llffhold:
            # 每llffhold张取1张将图片名存入到 测试相机图片名列表中
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]  # 遍历每个相机外参，获取所有图像的 图片名
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            # 从txt文件读取测试相机的 图片名
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        # 不评测，则所有相机均为训练相机，测试相机列表为空
        test_cam_infos = []

    # 存储原图片的文件夹名，默认为'images'，要从中读取图片
    reading_dir = "images" if images == None else images

    # 创建所有相机的Info类对象，存储到 cam_infos_unsorted列表中
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir),
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list)
    # 根据图片名称排序，以保证顺序一致性
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    # 如果是曝光补偿模式 或 该相机info不在测试相机列表中，则放入训练相机列表
    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    # 计算所有train相机的 中心点的tcw（"translate"），以及所有train相机到该点最大距离的1.1倍（"radius"）
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # 读取COLMAP生成的稀疏点云数据，优先从PLY文件读取，如果不存在，则尝试从BIN或TXT文件转换并保存为PLY格式
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)    # 从points3D.bin读取COLMAP产生的稀疏点云
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)

        storePly(ply_path, xyz, rgb)    # 转换成ply文件
    # 读取PLY格式的稀疏点云
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    # 组装场景信息，包括点云、训练用相机、测试用相机、场景归一化参数和点云文件路径
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
