import numpy as np
import argparse
import cv2
from joblib import delayed, Parallel
import json
# from read_write_model import *
from read_write_binary import *

def get_scales(key, cameras, images, points3d_ordered, args):
    """
    计算当前相机 估计的深度图 对齐到 COLMAP尺度的 scale 和 offset。返回一个字典，包含：图像名，scale，offset
        key：    当前相机索引
        cameras：存储 所有相机 内参信息的字典，每个元素包括：id(相机ID)、model(相机模型ID)、width、height、params(内参数组)
        images： 存储 所有相机 外参信息的字典，每个元素包括：id(图片ID)、qvec(W2C的旋转四元数)、tvec(W2C的平移向量)、camera_id(相机ID)、name(图像名)、xys(所有特征点的像素坐标)、point3D_ids(所有特征点对应3D点的ID，特征点没有生成3D点的ID则为-1)
        points3d_ordered：   按3D点的id顺序存储所有3D点的世界坐标 的数组，N 3
    """
    image_meta = images[key]    # 当前相机外参
    cam_intrinsic = cameras[image_meta.camera_id]   # 内参

    pts_idx = images_metas[key].point3D_ids # 当前相机所有特征点对应3D点的ID

    # 当前相机对应3D点的有效mask
    mask = pts_idx >= 0
    mask *= pts_idx < len(points3d_ordered)

    pts_idx = pts_idx[mask]     # 有效3D点 ID
    valid_xys = image_meta.xys[mask]    # 有效3D点对应特征点的像素坐标

    if len(pts_idx) > 0:
        pts = points3d_ordered[pts_idx] # 有效3D点的世界坐标
    else:
        pts = np.array([0, 0, 0])

    R = qvec2rotmat(image_meta.qvec)    # W2C的旋转矩阵
    pts = np.dot(pts, R.T) + image_meta.tvec    # 有效3D点在相机坐标系下的坐标

    # 相机坐标下 COLMAP计算的 稀疏3D点的 逆深度
    invcolmapdepth = 1. / pts[..., 2]

    # 读取png格式存储的 深度估计深度图（相对深度模式，估计的直接是 逆深度）
    n_remove = len(image_meta.name.split('.')[-1]) + 1
    invmonodepthmap = cv2.imread(f"{args.depths_dir}/{image_meta.name[:-n_remove]}.png", cv2.IMREAD_UNCHANGED)  # (H,W,3)

    if invmonodepthmap is None:
        return None
    
    if invmonodepthmap.ndim != 2:
        invmonodepthmap = invmonodepthmap[..., 0]   # (H,W)
    invmonodepthmap = invmonodepthmap.astype(np.float32) / (2**16)  # 归一化

    s = invmonodepthmap.shape[0] / cam_intrinsic.height # 原始深度图像 / 参与COLMAP计算的图像 的尺寸的比例

    maps = (valid_xys * s).astype(np.float32)   # COLMAP图像尺寸比例调整到原始深度图尺寸的 有效3D点对应特征点的像素坐标
    # 尺寸比例调整后 特征点有效mask，剔除 像素坐标超出范围 与 对应深度值<0 的点
    valid = (
        (maps[..., 0] >= 0) * 
        (maps[..., 1] >= 0) * 
        (maps[..., 0] < cam_intrinsic.width * s) * 
        (maps[..., 1] < cam_intrinsic.height * s) * (invcolmapdepth > 0))
    
    if valid.sum() > 10 and (invcolmapdepth.max() - invcolmapdepth.min()) > 1e-3:
        maps = maps[valid, :]
        # COLMAP逆深度 有效深度值
        invcolmapdepth = invcolmapdepth[valid]
        # 深度估计的逆深度 有效特征点 的深度值
        invmonodepth = cv2.remap(invmonodepthmap, maps[..., 0], maps[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[..., 0]
        
        ## Median / dev
        t_colmap = np.median(invcolmapdepth)    # COLMAP逆深度 中值
        s_colmap = np.mean(np.abs(invcolmapdepth - t_colmap))

        t_mono = np.median(invmonodepth)    # 深度估计逆深度 中值
        s_mono = np.mean(np.abs(invmonodepth - t_mono))

        scale = s_colmap / s_mono   # 深度图 到 COLMAP的映射关系
        offset = t_colmap - t_mono * scale
    else:
        scale = 0
        offset = 0
    return {"image_name": image_meta.name[:-n_remove], "scale": scale, "offset": offset}

if __name__ == '__main__':
    #! 生成DepthAnything-V2估计的深度图的参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default="../data/big_gaussians/standalone_chunks/campus") # COLMAP输出的sparse文件夹的父文件夹
    parser.add_argument('--depths_dir', default="../data/big_gaussians/standalone_chunks/campus/depths_any")    # DepthAnything-V2估计的深度图文件夹，保存的是png格式（DepthAnything相对深度模式估计的是逆深度）
    parser.add_argument('--model_type', default="bin")
    args = parser.parse_args()

    # 获取COLMAP估计的 相机内参、相机外参、点云数据
    cam_intrinsics, images_metas, points3d = read_model(os.path.join(args.base_dir, "sparse", "0"), ext=f".{args.model_type}")

    # 获取每个3D点的 id、世界坐标
    pts_indices = np.array([points3d[key].id for key in points3d])
    pts_xyzs = np.array([points3d[key].xyz for key in points3d])
    # 按3D点的id顺序存储所有3D点的世界坐标，N 3
    points3d_ordered = np.zeros([pts_indices.max()+1, 3])
    points3d_ordered[pts_indices] = pts_xyzs

    # 计算每张 估计的逆深度图的 sclae 和 offset（由深度图到COLMAP）
    # 并行计算 深度估计的逆深度 对齐到 COLMAP尺度的 scale和offset
    # depth_param_list = [get_scales(key, cam_intrinsics, images_metas, points3d_ordered, args) for key in images_metas]
    depth_param_list = Parallel(n_jobs=-1, backend="threading")(
        delayed(get_scales)(key, cam_intrinsics, images_metas, points3d_ordered, args) for key in images_metas
    )

    # 结果存储到一个字典中，key: 图像名，value: 一个字典，包含scale、offset
    depth_params = {
        depth_param["image_name"]: {"scale": depth_param["scale"], "offset": depth_param["offset"]}
        for depth_param in depth_param_list if depth_param != None
    }

    with open(f"{args.base_dir}/sparse/0/depth_params.json", "w") as f:
        json.dump(depth_params, f, indent=2)

    print(0)
