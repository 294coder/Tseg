import os
import random

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
from tqdm import tqdm

from utils.open3d_utils import array_to_pointcloud
from load_data import get_shapenet_data_path

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
color_all = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [.2, .4, .8], [1, 1, .3], [.2, .7, 1], [.5, 1, 1], [.4, .5, .6],
             [.6, .9, .2]]
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

label_dict = {'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4, 'Earphone': 5, 'Guitar': 6, 'Knife': 7,
              'Lamp': 8, 'Laptop': 9, 'Motorbike': 10, 'Mug': 11, 'Pistol': 12, 'Rocket': 13, 'Skateboard': 14,
              'Table': 15}
num_classes = 16
num_part = 50
batch_size = 16
lr = 1e-3
epoches = 20
PATH = '/home/Wangling/czh/PointNetWeight/seg.pth'
base_path = r'D:\codeblock\code\dachuang\Pointnet\data\shapenetcore_partanno_segmentation_benchmark_v0_normal'


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(x.shape)
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class Bottleneck(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample=None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer=None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        if downsample is not None:
            self.downsample = downsample
        else:
            self.downsample = nn.Conv1d(inplanes, planes * self.expansion, stride=stride, kernel_size=1)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class STN3d(nn.Module):
    def __init__(self, channel, device):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.device = device

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.to(self.device)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64, device='cuda:2'):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # self.se = SELayer(k)
        self.k = k

        self.device = device

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))

        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.to(self.device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        # x = self.se(x)
        return x


class SALayer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)  # b, n, c
        x_k = self.k_conv(x)  # b, c, n
        x_v = self.v_conv(x)
        energy = torch.bmm(x_q, x_k)  # b, n, n

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.bmm(x_v, attention)  # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


def feature_transform_reguliarzer(trans, device):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.to(device)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
    return loss


class get_model(nn.Module):
    def __init__(self, part_num=50, device='cuda:0', normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.part_num = part_num
        self.stn = STN3d(channel, device)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)  # 128->128
        self.conv4 = torch.nn.Conv1d(128, 512, 1)  # 128->512
        self.conv5 = torch.nn.Conv1d(512, 2048, 1)  # 512->2048
        # self.conv6 = torch.nn.Conv1d(2048, 4096, 1)  # 自己加的
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)  # 128
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)  # 2048
        # self.bn6 = nn.BatchNorm1d(4096)  # 自己加的
        self.fstn = STNkd(k=128, device=device)  # 128
        self.convs1 = torch.nn.Conv1d(4944, 256, 1)
        self.convs1_1 = torch.nn.Conv1d(4928, 256, 1)  # 4928
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, part_num, 1)  # 128
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)


    def forward(self, point_cloud, label=None, for_teeth=False):  # label: bs*1*bs
        B, D, N = point_cloud.size()
        trans = self.stn(point_cloud)
        point_cloud = point_cloud.transpose(2, 1)
        if D > 3:
            point_cloud, feature = point_cloud.split(3, dim=2)
        point_cloud = torch.bmm(point_cloud, trans)
        if D > 3:
            point_cloud = torch.cat([point_cloud, feature], dim=2)

        point_cloud = point_cloud.transpose(2, 1)

        out1 = F.relu(self.bn1(self.conv1(point_cloud)))  # 64
        # out1 = self.se1(out1)
        out2 = F.relu(self.bn2(self.conv2(out1)))  # 128
        # out2 = self.se2(out2)
        out3 = F.relu(self.bn3(self.conv3(out2)))  # 128
        # out3 = self.se3(out3)

        trans_feat = self.fstn(out3)  # 128
        x = out3.transpose(2, 1)
        net_transformed = torch.bmm(x, trans_feat)
        net_transformed = net_transformed.transpose(2, 1)

        out4 = F.relu(self.bn4(self.conv4(net_transformed)))  # 512
        # out4 = self.se4(out4)
        out5 = self.bn5(self.conv5(out4))  # 1024
        # out5 = self.se5(out5)  # 1024
        # out6 = self.bn6(self.conv6(out5))  # 2048
        # out6 = self.se6(out6)  # 2048
        out_max = torch.max(out5, 2, keepdim=True)[0]
        out_max = out_max.view(-1, 2048)

        if for_teeth:  # just a class, no need to concate
            expand = out_max.view(-1, 2048, 1).repeat(1, 1, N)
            concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)
            net = F.relu(self.bns1(self.convs1_1(concat)))  # 5056
            net = F.relu(self.bns2(self.convs2(net)))  # 256
            # net = net + out3.expand_as(net)  # residual
            net = F.relu(self.bns3(self.convs3(net)))  # 128
            # net = net + out2  # residual
            # ***
            # net = self.conv_a1(net)
            # net = self.conv_a2(net)
            # ***
            net = self.convs4(net)  # part_num
            net = net.transpose(2, 1).contiguous()
            net = F.log_softmax(net.view(-1, self.part_num), dim=-1)
            net = net.view(B, N, self.part_num)  # [B, N, 50]

            return net, trans_feat
        else:  # ModelNet40
            out_max = torch.cat([out_max, label.squeeze(1)], 1)
            expand = out_max.view(-1, 2048 + 16, 1).repeat(1, 1, N)  # 2048+batch_size
            concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)
            net = F.relu(self.bns1(self.convs1(concat)))
            net = F.relu(self.bns2(self.convs2(net)))
            net = F.relu(self.bns3(self.convs3(net)))

            net = self.convs4(net)
            net = net.transpose(2, 1).contiguous()
            net = F.log_softmax(net.view(-1, self.part_num), dim=-1)
            net = net.view(B, N, self.part_num)  # [B, N, 50]

            return net, trans_feat


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes, dtype=torch.int32)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001, device='cuda:5'):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale
        self.device = device

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat, self.device)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


class sampler(data.Sampler):
    """
    由于每个batch的点数可能不一致
    例如 len(b[0])=10220, len(b[1])=23300, len(b[2])=24000 , ...
    该sampler是为了将每个batch内的点数统一
    首先将batch里的样本按照点数从小到大排列
    返回排序之后的索引值
    """

    def __init__(self, data_source):
        super(sampler, self).__init__(data_source)
        self.x = data_source
        # y = data_source[1]
        self.lst = []
        for i in range(len(self.x)):
            self.lst.append(self.x[i].shape[0])
        self.idx = np.argsort(self.lst)  # 排序之后的索引

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.x[0])


def collate_fn(batch: list):
    """
    DataLoader中的最后一步
    对即将输出的batch进行操作
    这里将每一个batch的样本点数降到最少点数即min_num_pts
    :param batch: B*N*3 不同B中的N并不相同
    :return:
    """
    batch_size = len(batch)
    ret_cls = []
    for elem in batch:
        ret_cls.append(elem[2])
    ret_cls = torch.tensor(ret_cls)

    num_pts_lst = []
    for elem in batch:
        X, y = elem[0], elem[1]
        num_pts_lst.append(X.shape[0])
    sorted_lst = np.argsort(num_pts_lst)

    min_mun_pts = len(batch[sorted_lst[0]][0])

    temp_points = []
    temp_target = []
    for i in range(batch_size):
        X, y = batch[i][0], batch[i][1]
        torch.manual_seed(2021)
        X = X[torch.randperm(min_mun_pts)]
        torch.manual_seed(2021)
        y = y[torch.randperm(min_mun_pts)]
        temp_points.append(X)
        temp_target.append(y)
    ret_points = temp_points[0].unsqueeze(0)
    ret_target = temp_target[0].unsqueeze(0)

    for i in range(1, batch_size):
        cat_X = temp_points[i].unsqueeze(0)
        cat_y = temp_target[i].unsqueeze(0)
        ret_points = torch.cat([ret_points, cat_X], dim=0)
        ret_target = torch.cat([ret_target, cat_y], dim=0)
    return ret_points, ret_target, ret_cls


class data_provider(data.Dataset):
    def __init__(self, train: bool,
                 base_path: str = r'D:\codeblock\code\dachuang\shapenetcore_partanno_segmentation_benchmark_v0_normal',
                 with_normal: bool = False):
        super(data_provider, self).__init__()
        self.train = train
        self.with_normal = with_normal
        self.base_path = base_path
        ret_data_txt_path, ret_label_path, labels_dict = get_shapenet_data_path(self.train, 0.2, self.base_path)
        self.labels_dict = labels_dict
        self.all_path = ret_data_txt_path
        self.all_label_path = ret_label_path
        self.shuffle(self.all_path, self.all_label_path)

    def read_data(self, path, label_path):
        def _strip(data):
            return data.strip('\n').split()

        def _eval(data):
            return eval(data)

        f = open(os.path.join(self.base_path, label_path, path))
        raw = f.readlines()
        data = list(map(_strip, raw))
        ret_data = []
        ret_label = []
        for elem in data:
            l = list(map(_eval, elem))
            b = int(l[-1])
            if self.with_normal:
                l = l[:-1]
            else:
                l = l[:3]
            ret_data.append(l)
            ret_label.append(b)
        return ret_data, ret_label

    def shuffle(self, X, y):
        np.random.seed(2020)
        np.random.shuffle(X)
        np.random.seed(2020)
        np.random.shuffle(y)

    def __len__(self):
        return len(self.all_path)

    def __getitem__(self, idx):
        path = self.all_path[idx]
        label_path = self.all_label_path[idx]
        ret_data, ret_label = self.read_data(path, label_path)
        ret_data, ret_label = torch.tensor(ret_data), torch.tensor(ret_label)
        flag = ret_label[0].detach().item()
        label_name = seg_label_to_cat[flag]
        cls = label_dict[label_name]
        return ret_data, ret_label, cls


def check_load_data(data):
    """
    检测输入网络的数据是否正常
    :param data: B*N*3
    :return:
    """
    for pcd in data:
        pcd = array_to_pointcloud(pcd)
        o3d.visualization.draw_geometries([pcd])


def load_module_opti(path, part_num=2):
    d = torch.load(path, map_location='cpu')
    module_par = d['module_params']
    optim_par = d['optim_params']

    new_module = get_model(part_num=part_num, normal_channel=False)
    new_module.load_state_dict(module_par)

    optim = torch.optim.Adam(new_module.parameters(), lr=lr)
    optim.load_state_dict(optim_par)

    return new_module, optim


def test(classifier: nn.Module, test_iter):
    classifier.eval()
    all_num_pts = 0
    mean_correct = []
    for points, target, label in test_iter:
        all_num_pts = batch_size * len(target[0])
        label = label.view(num_classes, -1)
        points = points.transpose(2, 1)
        points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
        seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))
        seg_pred = seg_pred.contiguous().view(-1, num_part)
        target = target.view(-1, 1)[:, 0]
        pred_choice = seg_pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        mean_correct.append(correct.item() / all_num_pts)
    test_instance_acc = np.mean(mean_correct)
    print('-------Test accuracy is: %.5f' % test_instance_acc)


def predict(classifier: nn.Module, points, target, label):
    """
    justing seeing the predict labels and targets
    :param classifier:
    :param points:
    :param target: bs*numpts
    :param label: bs*1
    :return:
    """
    classifier.eval().cuda()
    label = label.view(num_classes, -1)
    points = points.transpose(2, 1)
    points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
    seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))
    seg_pred = seg_pred.contiguous().view(-1, num_part)
    target = target.view(-1, 1)[:, 0]
    pred_choice = seg_pred.data.max(1)[1]
    return pred_choice


def train(classifier, criterion):
    mean_correct = []
    classifier = classifier.cuda()
    criterion = criterion.cuda()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    p = data_provider(True, base_path=base_path)
    p_loader = data.DataLoader(p, batch_size, drop_last=True, collate_fn=collate_fn)  # train loader
    p2 = data_provider(False, base_path=base_path)
    p_loader2 = data.DataLoader(p2, batch_size, drop_last=True, collate_fn=collate_fn)  # test loader
    print(label_dict)
    for i in range(epoches):
        print('::epoch %d' % i)
        all_num_pts = 0
        for points, target, label in tqdm(p_loader):
            all_num_pts = batch_size * len(target[0])  # target [bs, num_pts]  label [bs]
            label = label.view(num_classes, -1)
            points = points.transpose(2, 1)
            optimizer.zero_grad()
            classifier = classifier.train()
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))
            seg_pred = seg_pred.contiguous().view(-1, num_part)  # seg_pred [bs*num_pts, num_parts]
            target = target.view(-1, 1)[:, 0]  # 拉成向量
            pred_choice = seg_pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / all_num_pts)
            loss = criterion(seg_pred, target, trans_feat)
            loss.backward()
            optimizer.step()
            print('loss is {:.4f}'.format(loss))
        train_instance_acc = np.mean(mean_correct)
        print('Train accuracy is: %.5f' % train_instance_acc)
        with torch.no_grad():
            test(classifier, criterion, p_loader2)

        save_dict = {
            'epoch': epoches,
            'module': classifier.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(save_dict, PATH)


if __name__ == '__main__':
    # classifier, _ = load_module_opti(r'D:\codeblock\code\dachuang\ModuleWeight\seg.pth')
    # classifier = get_model(normal_channel=False).cuda()
    # batch_size_pred = 16
    # p2 = data_provider(False)
    # p_loader2 = data.DataLoader(p2, batch_size_pred, drop_last=True, collate_fn=collate_fn)  # test loader
    # criterion = get_loss().cuda()
    # train(classifier, criterion)
    loss = nn.NLLLoss()
    a = torch.rand([10, 3, 1000])
    y = torch.randint(0, 4, (10, 1000))
    module = get_model(part_num=4, normal_channel=False, device='cpu')
    x, _ = module(a, for_teeth=True)
    print(x.shape)
    y = y.view(-1, 1)[:, 0]
    x = x.contiguous().view(-1, 4)
    l = loss(x, y)
    l.backward()
    pass
