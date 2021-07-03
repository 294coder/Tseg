import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import open3d as o3d
import numpy as np
import torch.utils.data as data
from load_data import get_noted_teeth_data as gntd
from load_data import get_modelnet_data as gmd
import matplotlib.pyplot as plt


def farthest_point_sample(xyz, npoint):
    """
    最远点采样是Set Abstraction模块中较为核心的步骤，
    其目的是从一个输入点云中按照所需要的点的个数npoint采样出足够多的点，
    并且点与点之间的距离要足够远。
    最后的返回结果是npoint个采样点在原始点云中的索引

    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples 采样点数
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)  # B*npoints
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        # 更新第i个最远点
        centroids[:, i] = farthest  # centroids[:i]: (B,)    farthest: (B,)
        # 取出这个最远点的xyz坐标
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)  # batch_indices: (B,)    xyz: (B,N,C) C=3
        # 计算点集中的所有点到这个最远点的欧式距离
        dist = torch.sum((xyz - centroid) ** 2, -1)  # dist: (B, N)   -1表示在通道层上求最大
        # 更新distances，记录样本中每个点距离所有已出现的采样点的最小距离
        mask = dist < distance  # distance: (B,N)
        distance[mask] = dist[mask]  # mask: (B,N) elem is type(Bool)
        # 从更新后的distances矩阵中找出距离最远的点，作为最远点用于下一轮迭代
        farthest = torch.max(distance, -1)[1]
        # [1] means indices     [0] means max distance      which are function max()'s return
    return centroids


def square_distance(src, dst):
    """
    该函数主要用来在ball query过程中确定每个点距离采样点的距离
    函数输入是两组点，N为第一组点的个数，M为第二组点的个数，C为输入点的通道数（如果是xyz时C=3）
    返回的是两组点之间两两的欧几里德距离
    即N × M的矩阵
    由于在训练中数据通常是以Mini-Batch的形式输入的，所以有一个Batch数量的维度为B
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
	     = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst


    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # 2*(xn * xm + yn * ym + zn * zm)
    dist += torch.sum(src ** 2, -1).view(B, N, 1)  # xn*xn + yn*yn + zn*zn
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)  # xm*xm + ym*ym + zm*zm
    return dist


def index_points(points, idx):
    """
    按照输入的点云数据和索引返回由索引的点云数据

    eg.
    points为B × 2048 × 3的点云，idx为[ 1 , 333 , 1000 , 2000 ]
    则返回B个样本中每个样本的第1,333,1000,2000个点组成的B × 4 × 3 的点云集
    当然如果idx为一个[ B , D 1 , . . . D N ]维度的，
    则它会按照idx中的维度结构将其提取成[ B , D 1 , . . . D N , C ]

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, D1,...DN]
    Return:
        new_points:, indexed points data, [B, D1,...DN, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)  # 去除batch_size     并将B之后的值置为1
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1  # 去除batch_size之后的     并将B的值置为1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    query_ball_point函数用于寻找球形领域中的点。
    输入中radius为球形领域的半径，nsample为每个领域中要采样的点，
    new_xyz为S个球形领域的中心（由最远点采样在前面得出），xyz为所有的点云；
    输出为每个样本的每个球形领域的nsample个采样点集的索引[B,S,nsample]

    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape  # S个中心点
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    # sqrdists: [B, S, N] 记录中心点与所有点之间的欧几里德距离
    sqrdists = square_distance(new_xyz, xyz)
    # 找到所有距离大于radius^2的，其group_idx直接置为N；其余的保留原来的值
    group_idx[sqrdists > radius ** 2] = N
    # 做升序排列，前面大于radius^2的都是N，会是最大值，所以会直接在剩下的点中取出前nsample个点
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]  # 大于阈值的点升序排在最后 取[0] 得到index 再取前nsample个
    # 考虑到有可能前nsample个点中也有被赋值为N的点（即球形区域内不足nsample个点），这种点需要舍弃，直接用第一个点来代替即可
    # group_first: [B, S, k]， 实际就是把group_idx中的第一个点的值复制为了[B, S, K]的维度，便利于后面的替换
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    # 找到group_idx中值等于N的点
    mask = group_idx == N
    # 将这些点的值替换为第一个点的值
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Input:
        npoint: Number of point for FPS
        radius: Radius of ball query
        nsample: Number of point for each ball query
        xyz: Old feature of points position data, [B, N, C]
        points: New feature of points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, C]
        new_points: sampled points data, [B, npoint, nsample, C+D]
    """
    B, N, C = xyz.shape
    # 从原点云中挑出最远点采样的采样点为new_xyz
    new_xyz = index_points(xyz, farthest_point_sample(xyz, npoint))
    # idx:[B, npoint, nsample] 代表npoint个球形区域中每个区域的nsample个采样点的索引
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    # grouped_xyz:[B, npoint, nsample, C]
    grouped_xyz = index_points(xyz, idx)
    # grouped_xyz减去采样点即中心值
    grouped_xyz -= new_xyz.view(B, npoint, 1, C)
    # 如果每个点上面有新的特征的维度，则用新的特征与旧的特征拼接，否则直接返回旧的特征
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        '''
        首先先通过sample_and_group的操作形成局部的group，然后对局部的group中的每一个点做MLP操作，
        最后进行局部的最大池化，得到局部的全局特征
        Input:
            npoint: Number of point for FPS sampling
            radius: Radius for ball query
            nsample: Number of point for each ball query
            in_channel: the dimention of channel
            mlp: A list for mlp input-output channel, such as [64, 64, 128]
            group_all: bool type for group_all or not
        '''
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)  # [B,N,C]
        if points is not None:
            points = points.permute(0, 2, 1)
        # 形成group
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        # 利用1x1的2d的卷积相当于把每个group当成一个通道，共npoint个通道，对[C+D, nsample]的维度上做逐像素的卷积，结果相当于对单个C+D维度做1d的卷积
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        # 对每个group做一个max pooling得到局部的全局特征
        new_points = torch.max(new_points, 2)[0]  # coordinate
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    """
    大部分的形式都与普通的SA层相似，
    但是这里radius_list输入的是一个list例如[0.1,0.2,0.4]，
    对于不同的半径做ball query，最终将不同半径下的点点云特征保存在new_points_list中，
    再最后拼接到一起
    """

    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):  # has channels in mlp_list[i]
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3  # why plus 3?
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)  # change dimension
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):  # every radius in radius_list create a points group
            # create point group[i]
            K = self.nsample_list[i]
            # same as func: sample_and_group()
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)  # 在D层上拼接
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    """
    FeaturePropagation层的实现主要通过线性差值与MLP堆叠完成。
    当点的个数只有一个的时候，采用repeat直接复制成N个点；
    当点的个数大于一个的时候，采用线性差值的方式进行上采样，
    再对上采样后的每一个点都做一个MLP，同时拼接上下采样前相同点个数的SA层的特征
    """

    # for segmentation

    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)  # func: sort() return values and indices
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            dists[dists < 1e-10] = 1e-10  # small enough
            weight = 1.0 / dists  # [B, N, 3]
            weight = weight / torch.sum(weight, dim=-1).view(B, N, 1)  # [B, N, 3]
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class get_model(nn.Module):
    def __init__(self, num_classes, normal_channel=False):
        super(get_model, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=6 + additional_channel,
                                          mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3,
                                          mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + 6 + additional_channel, mlp=[128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        # Set Abstraction layers
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)

        return total_loss


def _draw_pic_before_train(train_iter_X, train_iter_y):  # 验证输入数据的正确性 仅在训练时调用
    from get_division_polylines import array_to_pointcloud
    # 画出传入PointNet++的数据图像
    for X, y in zip(train_iter_X, train_iter_y):
        print(X.shape)
        X = X.view(3, 3, -1)
        y = y.view(3, -1)
        for i in range(2):
            X1 = X[i].view(-1, 3)
            X1 = X1.contiguous().detach().cpu().numpy()
            pcd = array_to_pointcloud(X1)
            o3d.visualization.draw_geometries([pcd])
            line = []
            for j in range(len(y[i])):
                if y[i][j] == 1:
                    line.append(X1[j])
            pcd1 = array_to_pointcloud(np.array(line))
            o3d.visualization.draw_geometries([pcd1])


if __name__ == "__main__":
    from processing_data import convert_to_datasets_four_class

    module = get_model(4)
    X = torch.randn(10, 3, 8192)
    x, l4_points = module(X)
    print(x.shape)
    print(l4_points.shape)
