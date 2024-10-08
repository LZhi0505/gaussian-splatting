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
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

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

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    '''
        cam_extrinsics: 存储所有相机的 外参的字典，每个元素包括：id(图片ID)、qvec(W2C的旋转四元数)、tvec(W2C的平移向量)、camera_id(相机ID)、name(图像名)、xys(所有特征点的像素坐标)、point3D_ids(所有特征点对应3D点的ID，特征点没有生成3D点的ID则为-1)
        cam_intrinsics: 存储所有相机的 内参的字典，每个元素包括：id(相机ID)、model(相机模型ID)、width、height、params(内参数组)
        images_folder: 保存RGB图的文件夹路径
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

        # PIL.Image读取的为RGB格式，OpenCV读取的为BGR格式
        image_path = os.path.join(images_folder, extr.name)
        if not os.path.exists(image_path):
            continue

        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        # 创建相机信息类CameraInfo对象 (包含R、T、FovY、FovX、图像数据image、image_path、image_name、width、height)，并添加到列表cam_infos中
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
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

def readColmapSceneInfo(path, images, eval, llffhold=8):
    '''
    加载COLMAP的结果中的二进制相机外参文件imags.bin 和 内参文件cameras.bin
        path:   source_path
        images: "images"
        eval:   是否为eval模式
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

    # 存储原图片的文件夹名，默认为'images'，要从中读取图片
    reading_dir = "images" if images == None else images

    # 创建所有相机的Info类对象，存储到 cam_infos_unsorted列表中
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    # 根据图片名称排序，以保证顺序一致性
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    # 根据是否eval，将相机分为训练集和测试集
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        # 如果不是评估模式，所有相机均为训练相机，测试相机列表为空
        train_cam_infos = cam_infos
        test_cam_infos = []

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
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
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

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
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
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
