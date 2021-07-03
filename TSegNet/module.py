from Pointnet.models.pointnet_util import PointNetSetAbstraction, PointNetFeaturePropagation, PointNetSetAbstractionMsg
import torch
import torch.nn as nn
import torch.functional as F


class Encoder(nn.Module):
    def __init__(self, subsample=False):
        super(Encoder, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.025, 0.05], [16, 32], 3,
                                             [[3, 32, 32], [3, 32, 32]])
        self.sa2 = PointNetSetAbstractionMsg(512, [0.05, 0.1], [16, 32], 64,
                                             [[64, 64, 128], [64, 64, 128]])
        self.sa3 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 256,
                                             [[256, 196, 256], [256, 196, 256]])

        self.subsample = subsample
        if subsample:
            self.sa4 = PointNetSetAbstraction(npoint=256, radius=0.1, nsample=128, in_channel=512 + 3,
                                              mlp=[512, 256],group_all=False)
            self.conv = nn.Sequential(nn.Conv1d(512, 3, kernel_size=1),
                                      nn.BatchNorm1d(3),
                                      nn.ReLU()
                                      )

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        if self.subsample:
            l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
            subsample = torch.cat([l4_points, l4_xyz], dim=1)  # 256+3
            return subsample

        return l1_xyz, l2_xyz, l3_xyz, l3_points, l2_points, l1_points


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fp1 = PointNetFeaturePropagation(768, mlp=[768, 256, 256])
        self.fp2 = PointNetFeaturePropagation(320, mlp=[320, 128, 128])
        self.fp3 = PointNetFeaturePropagation(128 + 3, mlp=[128 + 3, 64, 3])

    def forward(self, xyz, l1_xyz, l2_xyz, l3_xyz, l3_points, l2_points, l1_points):
        l0_xyz = xyz
        l2_points = self.fp1(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp3(l0_xyz, l1_xyz, xyz, l1_points)
        return l0_points


class Backbond(nn.Module):
    def __init__(self):
        super(Backbond, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, xyz):
        l1_xyz, l2_xyz, l3_xyz, l3_points, l2_points, l1_points = self.encoder(xyz)
        l0_points = self.decoder(a, l1_xyz, l2_xyz, l3_xyz, l3_points, l2_points, l1_points)
        return l0_points


class Centroid(nn.Module):
    def __init__(self):
        super(Centroid, self).__init__()
        self.encoder = Encoder(subsample=True)
        self.xyz_conv = nn.Sequential(nn.Conv1d(256 + 3, 256, kernel_size=1),
                                      nn.BatchNorm1d(256),
                                      nn.LeakyReLU(),
                                      nn.Conv1d(256, 3, kernel_size=1)
                                      )

    def forward(self, xyz):
        encode_points = self.encoder(xyz)
        centroids = self.xyz_conv(encode_points)
        return centroids


if __name__ == '__main__':
    # encoder = Encoder(subsample=True)
    # decoder = Decoder()
    c=Centroid()
    a = torch.rand((16, 3, 8192))
    # l1_xyz, l2_xyz, l3_xyz, l3_points, l2_points, l1_points = encoder(a)
    centroids=c(a)
    pass
    # decoder(a, l1_xyz, l2_xyz, l3_xyz, l3_points, l2_points, l1_points)
