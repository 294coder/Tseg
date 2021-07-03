import numpy as np
from utils.open3d_utils import array_to_pointcloud
import open3d as o3d
from typing import Union
import torch.nn.functional as F
import torch
import torch.nn as nn
import random
import warnings


# 5/26
def centroids_clustering(centroid, src, cluster_num=None, **kwargs):
    # kwargs contains "factor"
    if len(kwargs) != 0:
        factors = kwargs['factors']
    else:
        factors = 0.4
    assert 0 < factors < 1
    if cluster_num is None:
        cluster_num = int(len(src) * factors)

    offset = src - centroid
    dist = np.sum(offset ** 2, axis=1)
    argdist = dist.argsort()
    mask = argdist[:cluster_num]
    cluster = src[mask]
    return cluster
    # test
    # cluster_pcd = array_to_pointcloud(cluster).paint_uniform_color([0, 0, 1])
    # t_remove_cluster = np.delete(t, mask, axis=0)
    # t_remove_cluster_pcd = array_to_pointcloud(t_remove_cluster)
    # t_remove_cluster_pcd = array_to_pointcloud(t_remove_cluster).paint_uniform_color([1, 0, 0])
    # o3d.visualization.draw_geometries([t_remove_cluster_pcd, cluster_pcd])


def cluster_by_miny(src, miny):
    mask = src[:, 1] > miny
    cluster = mask[miny]
    return cluster


def make_centroids(*teeth):
    '''
    t in teeth:
    t shape is [N,3]
    N is pts
    3 is xyz
    :param teeth: list or tuple, usually 4
    :return: list, every centroid of those teeth
    '''
    centroids = []
    for t in teeth:
        centroids.append(t.mean(axis=0))
    return centroids
    # test
    # a = np.random.randint(0, 10, (10, 3))
    # make_centroids(a)


def upsample_teeth(teeth: Union[np.ndarray, torch.Tensor], size=8192):
    t_sz = teeth.shape
    if isinstance(teeth, torch.Tensor):
        teeth = teeth.cpu()
    if isinstance(teeth, np.ndarray):
        teeth = torch.from_numpy(teeth)

    if len(t_sz) > 2:  # [batch_size,pts,3] or [batch_size,3,pts]
        if t_sz[1] == 3:  # without norm
            teeth = teeth.transpose(1, 2)  # [batch_size,pts,3]
        if max(t_sz) > size:  # pts>size
            perm = torch.randperm(size)
            teeth = teeth[:, :perm, :][:, :size, :]  # size
            return teeth
        teeth_points = F.interpolate(teeth[None, :, :, :], size=[size, 3]).squeeze()
    else:  # [pts,3]
        if t_sz[0] == 3:  # without norm
            teeth = teeth.transpose(0, 1)  # [batch_size,pts,3]
        if max(t_sz) > size:  # pts>size
            perm = torch.randperm(size)
            teeth = teeth[perm, :][:size, :]  # size
            return teeth
        teeth_points = F.interpolate(teeth[None, None, :, :], size=[size, 3]).squeeze()

    return teeth_points


def density_adaptive_downsampling(teeth: Union[o3d.geometry.PointCloud, np.ndarray, torch.Tensor], size: int = 8192,
                                  voxel_size=0.5):
    sz = [size, 3]
    if isinstance(teeth, torch.Tensor):
        teeth = np.array(teeth)
    if isinstance(teeth, np.ndarray):
        teeth = array_to_pointcloud(teeth)
    teeth = teeth.voxel_down_sample(voxel_size=voxel_size)
    teeth_points = np.asarray(teeth.points)
    if teeth_points.shape[0] < size:
        warnings.warn("interpolate points, may cause label incorrect")
        teeth_points = torch.from_numpy(teeth_points).float()
        teeth_points = F.interpolate(teeth_points[None, None, :, :], size=sz).squeeze()
        teeth_points = np.array(teeth_points)
    # else:
    #     np.random.seed(2021)
    #     np.random.shuffle(teeth_points)
    #     teeth_points = teeth_points[:size, :]
    return teeth_points
    # test
    # t = np.loadtxt("D:/desktop/all.txt")
    # t = array_to_pointcloud(t)
    # o3d.visualization.draw_geometries([t])
    # t_dsp = density_adaptive_downsampling(t)
    # t_dsp = array_to_pointcloud(t_dsp)
    # o3d.visualization.draw_geometries([t_dsp])


def voxel_filter(point_cloud, leaf_size=0.01, filter_mode='random'):
    filtered_points = []
    # filtered_labels = []
    # step1 计算边界点
    x_max, y_max, z_max = np.amax(point_cloud, axis=0)  # 计算 x,y,z三个维度的最值
    x_min, y_min, z_min = np.amin(point_cloud, axis=0)
    # step2 确定体素的尺寸
    size_r = leaf_size
    # step3 计算每个 volex的维度
    Dx = (x_max - x_min) / size_r
    Dy = (y_max - y_min) / size_r
    Dz = (z_max - z_min) / size_r
    # step4 计算每个点在volex grid内每一个维度的值
    h = list()
    for i in range(len(point_cloud)):
        hx = np.floor((point_cloud[i][0] - x_min) / size_r)
        hy = np.floor((point_cloud[i][1] - y_min) / size_r)
        hz = np.floor((point_cloud[i][2] - z_min) / size_r)
        h.append(hx + hy * Dx + hz * Dx * Dy)
    # step5 对h值进行排序
    h = np.array(h)
    h_indice = np.argsort(h)  # 提取索引
    h_sorted = h[h_indice]  # 升序
    # label = label[h_indice]
    count = 0  # 用于维度的累计
    # 将h值相同的点放入到同一个grid中，并进行筛选
    for i in range(len(h_sorted) - 1):  # 0-19999个数据点
        if h_sorted[i] == h_sorted[i + 1]:  # 当前的点与后面的相同，放在同一个volex grid中
            continue
        else:
            '''弃用均值滤波'''
            if filter_mode == "centroid":  # 均值滤波
                point_idx = h_indice[count: i + 1]
                filtered_points.append(np.mean(point_cloud[point_idx], axis=0))  # 取同一个grid的均值
                count = i

            elif filter_mode == "random":  # 随机滤波
                point_idx = h_indice[count: i + 1]
                random.seed(2021)
                random_points = random.choice(point_cloud[point_idx])
                random.seed(2021)
                random_label = random.choice(label[point_idx])
                filtered_points.append(random_points)
                # filtered_labels.append(random_label)
                count = i

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    # filtered_labels = np.array(filtered_labels, dtype=np.int)
    return filtered_points


if __name__ == "__main__":
    from load_data import read_teeth_data, read_teeth_data_four_class
    from processing_data import convert_to_datasets_four_class
    from torch.utils.data import DataLoader

    path = r'D:\desktop\teeth_program_read\rotate（41-50）'
    datasets = convert_to_datasets_four_class(path=path, scale=False)
    dataloader = DataLoader(datasets, batch_size=1, shuffle=False, drop_last=False)
    for X, y in dataloader:
        X = np.array(X.squeeze())
        pcd = array_to_pointcloud(X)
        mask = np.array(y == 1).squeeze()
        t1 = X[mask]
        t1 = array_to_pointcloud(t1)
        o3d.visualization.draw_geometries([t1])
