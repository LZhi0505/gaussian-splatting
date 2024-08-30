import numpy as np
import argparse
import cv2
from joblib import delayed, Parallel
import json
# from read_write_model import *
from read_write_binary import *

def get_scales(key, cameras, images, points3d_ordered, args):
    """

        key：    当前相机索引
        cameras：存储 所有相机 内参信息的字典，每个元素包括：id 相机ID、model 相机模型ID、width、height、params 内参数组
        images： 存储 所有相机 外参信息的字典，每个元素包括：id 图片ID、qvec W2C的旋转四元数、tvec W2C的平移向量、camera_id 相机ID、name 图像名、xys 所有特征点的像素坐标、point3D_ids 所有特征点对应3D点的ID（特征点没有生成3D点的ID则为-1）
        points3d_ordered：   存储 所有点云信息 的字典，每个元素包括：id 3D点ID、xyz 世界坐标、rgb 颜色值、error、image_ids、point2D_idxs
    """
    image_meta = images[key]    # 当前相机外参
    cam_intrinsic = cameras[image_meta.camera_id]

    pts_idx = images_metas[key].point3D_ids # 当前相机所有特征点对应3D点的ID

    mask = pts_idx >= 0     # 存在映射3D点的特征点 mask
    mask *= pts_idx < len(points3d_ordered)

    pts_idx = pts_idx[mask]     # 当前相机的有效3D点ID
    valid_xys = image_meta.xys[mask]    # 有效特征点的像素坐标

    if len(pts_idx) > 0:
        pts = points3d_ordered[pts_idx] # 有效3D点
    else:
        pts = np.array([0, 0, 0])

    R = qvec2rotmat(image_meta.qvec)    # W2C的旋转矩阵
    pts = np.dot(pts, R.T) + image_meta.tvec    # 有效3D点在相机坐标系下的坐标

    invcolmapdepth = 1. / pts[..., 2]   #

    # 读取估计的深度图片 png
    n_remove = len(image_meta.name.split('.')[-1]) + 1
    invmonodepthmap = cv2.imread(f"{args.depths_dir}/{image_meta.name[:-n_remove]}.png", cv2.IMREAD_UNCHANGED)  # (H,W,3)
    
    if invmonodepthmap is None:
        return None
    
    if invmonodepthmap.ndim != 2:
        invmonodepthmap = invmonodepthmap[..., 0]   # (H,W)

    invmonodepthmap = invmonodepthmap.astype(np.float32) / (2**16)
    s = invmonodepthmap.shape[0] / cam_intrinsic.height

    maps = (valid_xys * s).astype(np.float32)   # 有效特征点的像素坐标
    valid = (
        (maps[..., 0] >= 0) * 
        (maps[..., 1] >= 0) * 
        (maps[..., 0] < cam_intrinsic.width * s) * 
        (maps[..., 1] < cam_intrinsic.height * s) * (invcolmapdepth > 0))
    
    if valid.sum() > 10 and (invcolmapdepth.max() - invcolmapdepth.min()) > 1e-3:
        maps = maps[valid, :]
        invcolmapdepth = invcolmapdepth[valid]
        invmonodepth = cv2.remap(invmonodepthmap, maps[..., 0], maps[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[..., 0]
        
        ## Median / dev
        t_colmap = np.median(invcolmapdepth)
        s_colmap = np.mean(np.abs(invcolmapdepth - t_colmap))

        t_mono = np.median(invmonodepth)
        s_mono = np.mean(np.abs(invmonodepth - t_mono))
        scale = s_colmap / s_mono
        offset = t_colmap - t_mono * scale
    else:
        scale = 0
        offset = 0
    return {"image_name": image_meta.name[:-n_remove], "scale": scale, "offset": offset}

if __name__ == '__main__':
    #! 生成DepthAnything-V2估计的深度图的参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default="../data/big_gaussians/standalone_chunks/campus") # COLMAP输出的sparse文件夹的父文件夹
    parser.add_argument('--depths_dir', default="../data/big_gaussians/standalone_chunks/campus/depths_any")    # DepthAnything-V2估计的深度图文件夹，保存的是png格式
    parser.add_argument('--model_type', default="bin")
    args = parser.parse_args()

    # 获取COLMAP估计的 相机内参、相机外参、稀疏点云。对应
    cam_intrinsics, images_metas, points3d = read_model(os.path.join(args.base_dir, "sparse", "0"), ext=f".{args.model_type}")

    # 获取每个点云的 索引、中心的世界坐标、
    pts_indices = np.array([points3d[key].id for key in points3d])
    pts_xyzs = np.array([points3d[key].xyz for key in points3d])
    points3d_ordered = np.zeros([pts_indices.max()+1, 3])
    points3d_ordered[pts_indices] = pts_xyzs

    # 计算估计的 逆深度图的sclae和offset（并行处理每个相机）
    # depth_param_list = [get_scales(key, cam_intrinsics, images_metas, points3d_ordered, args) for key in images_metas]
    depth_param_list = Parallel(n_jobs=-1, backend="threading")(
        delayed(get_scales)(key, cam_intrinsics, images_metas, points3d_ordered, args) for key in images_metas
    )

    depth_params = {
        depth_param["image_name"]: {"scale": depth_param["scale"], "offset": depth_param["offset"]}
        for depth_param in depth_param_list if depth_param != None
    }

    with open(f"{args.base_dir}/sparse/0/depth_params.json", "w") as f:
        json.dump(depth_params, f, indent=2)

    print(0)
