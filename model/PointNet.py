import time

import os
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data as data
from torch.autograd import Variable
import matplotlib.pyplot as plt
from get_division_polylines import array_to_pointcloud
from ModelNetDataLoader import ModelNetDataLoader
import provider

import torchvision.models as models





class STN3d(nn.Module):
    def __init__(self, channel):
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
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
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

        self.k = k

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
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            x, feature = x.split(3, dim=2)
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        if torch.cuda.device_count() > 1:
            I = I.to('cuda:1')
        else:
            I = I.to('cuda:0')
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=False):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_regularizer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


class data_provider(data.Dataset):

    def _strip(self, i):
        return i.strip()

    def __init__(self, train, shuffle=True):
        super(data_provider, self).__init__()
        self.BASE_PATH = r'D:\codeblock\code\dachuang\modelnet40_normal_resampled'
        self.NAME_PATH = r'D:\codeblock\code\dachuang\modelnet40_normal_resampled\modelnet40_shape_names.txt'
        self.TRAIN_PATH = r'D:\codeblock\code\dachuang\modelnet40_normal_resampled\modelnet40_train.txt'
        self.TEST_PATH = r'D:\codeblock\code\dachuang\modelnet40_normal_resampled\modelnet40_test.txt'

        f_names = open(self.NAME_PATH)
        f_train = open(self.TRAIN_PATH)
        f_test = open(self.TEST_PATH)

        self.names_dict = dict()
        self._labels = list(map(self._strip, f_names.readlines()))
        for num, i in enumerate(self._labels, 0):
            self.names_dict[i] = num
        if train:
            self.data_path = list(map(self._strip, f_train.readlines()))
        else:
            self.data_path = list(map(self._strip, f_test.readlines()))

        if shuffle:
            self._shuffle()

    def _downsample(self, data):
        np.random.shuffle(data)
        return data[:1024, :]

    def _get_label(self, label_path):
        label_split = label_path.split('_')
        if len(label_split) == 2:
            return label_split[0]
        elif len(label_split) == 3:
            return label_split[0] + '_' + label_split[1]

    def _read(self, path):
        label = self._get_label(path)
        encode = self.names_dict[label]
        now_path = os.path.join(self.BASE_PATH, label, path)
        data = np.loadtxt(now_path, delimiter=',').astype(np.float32)[:, :3]
        return data, encode

    def __len__(self):
        return len(self.data_path)

    def _shuffle(self):
        np.random.shuffle(self.data_path)
        np.random.shuffle(self.data_path)

    def get_labels(self) -> dict:
        return self.names_dict

    def __getitem__(self, idx):
        fn = self.data_path[idx] + '.txt'
        data, encode = self._read(fn)
        data = self._downsample(data)
        return data, encode


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


def up_sample(input, size):
    """
    上采样
    :param input: 传入样本点
    :param size: 输出大小
    :return: 输出点
    """
    return F.interpolate(input, size, mode='bilinear', align_corners=True)


def collect_fn(batch):
    """
    :param batch: data batch
    :return:
    """
    # min_len = batch[0].shape[0]
    # max_len = batch[-1].shape[0]
    ds_pts = 1024  # 1024个点
    num_pts = batch[0].shape[0]
    new_batch = batch[-1].unsqueeze(0)
    new_batch = new_batch[:, torch.randperm(new_batch.shape[1]), :][:, :ds_pts, :].unsqueeze(0)
    for i in range(0, len(batch) - 1):
        now_elem: torch.Tensor = batch[i].view(1, num_pts,
                                               3).contiguous()  # torch.reshape(batch[i], (1, 1, num_pts, 3))
        # size = (ds_pts, 3)
        # now_elem = up_sample(now_elem, size)
        now_elem = now_elem[:, torch.randperm(now_elem.shape[-2]), :][:, :ds_pts, :].unsqueeze(0)  # 随机下采样
        new_batch = torch.cat([new_batch, now_elem])

    return new_batch.squeeze(1)


# 暂时没用
def rate_correct(y_pred, y, batch_size):
    count = 0
    correct = (y_pred == y.squeeze().T)
    for i in correct:
        if i:
            count += 1
    print("train correct rate is %.3f" % (count / batch_size))


def load_make_module(module_path):
    module = torch.load(module_path)
    classifier = get_model(k=40)
    classifier.load_state_dict(module['module'])
    return classifier


# 暂时用不到
def predict(test_datasets: data_provider, test_iter_X):
    net = load_make_module(module_path=r'D:\codeblock\code\dachuang\ModuleWeight\weight_2021.3.24_22_46.pt')
    net.cuda().eval()
    labels_dict = test_datasets.get_labels()
    for X, y in test_iter_X:
        X1 = X.transpose(2, 1).cuda()
        y_hat, _ = net(X1)
        y_pred = y_hat.argmax(dim=1)
        print(y)
        print(y_pred)
        for y_p in zip(y_pred):
            for k, v in labels_dict.items():
                if v == y_p:
                    print(k, end=' ')

        for y_t in zip(y):
            for k, v in labels_dict.items():
                if v == y_t:
                    print(k)

        check_load_data(X)


def draw_loss(all_loss, epoches, i):
    # 画出传入PointNet的数据图像
    x = [i for i in range(epoches)]
    import matplotlib.pyplot as plt

    plt.plot(x, all_loss, '-')
    plt.xlabel('epoches')
    plt.ylabel('loss')
    if i % 5 == 0:
        plt.savefig("./loss/loss_" + str(i) + ".jpg")


def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    for points, target in tqdm(loader, total=len(loader)):
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def check_load_data(data):
    """
    检测输入网络的数据是否正常
    :param data: B*N*3
    :return:
    """
    for pcd in data:
        pcd = array_to_pointcloud(pcd)
        o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    from tqdm import tqdm

    gm = time.gmtime()
    t = str(gm.tm_year) + '.' + str(gm.tm_mon) + '.' + str(gm.tm_mday) + '_' + str(gm.tm_hour + 8) + ':' + str(
        gm.tm_min)

    DATA_PATH = './modelnet40_normal_resampled/'
    SAVE_PATH = '/home/Wangling/czh/PointNetWeight/weight_' + t + '.pt'
    NAME_PATH = r'D:\codeblock\code\dachuang\modelnet40_normal_resampled\modelnet40_shape_names.txt'
    TRAIN_PATH = r'D:\codeblock\code\dachuang\modelnet40_normal_resampled\modelnet40_train.txt'
    TEST_PATH = r'D:\codeblock\code\dachuang\modelnet40_normal_resampled\modelnet40_test.txt'
    print(SAVE_PATH)
    train_dataset = data_provider(train=True)
    test_dataset = data_provider(train=False)

    labels_dict = train_dataset.get_labels()

    trainDataLoader = data.DataLoader(train_dataset, batch_size=16, drop_last=True)
    testDataLoader = data.DataLoader(test_dataset, batch_size=16, drop_last=True)
    module = load_make_module(r'D:\codeblock\code\dachuang\ModuleWeight\weight_2021.3.24_22_46.pt')
    # instance_acc, class_acc = test(module.cuda(), testDataLoader)
    # print(instance_acc)
    predict(test_dataset, testDataLoader)

    '''
    # TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=1024, split='train',
    #                                    normal_channel=False)
    # TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=1024, split='test',
    #                                   normal_channel=False)
    # trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=16, shuffle=True,
    #                                               num_workers=4)
    # testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=16, shuffle=False, num_workers=4)

    epoches = 200
    classifier = get_model(k=40).cuda()
    criterion = get_loss().cuda()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    global train_instance_acc, test_instance_acc

    for epoch in range(epoches):
        mean_correct = []
        print("::epoch %d" % (epoch + 1))
        # for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
        for points, target in tqdm(trainDataLoader, total=len(trainDataLoader), smoothing=.9):
            # points, target = data
            # points = points.data.numpy()
            # points = provider.random_point_dropout(points)
            # points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            # points = torch.Tensor(points)
            # target = target[:, 0]

            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()

            classifier = classifier.train()
            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()

        train_instance_acc = np.mean(mean_correct)
        print("train_acc is %.3f" % train_instance_acc)

        with torch.no_grad():
            test_instance_acc, class_acc = test(classifier.eval(), testDataLoader)
            print("test_acc is {:.3f}, class_acc is {:.3f}".format(test_instance_acc, class_acc))

        save_dict = {
            'epoches': epoches,
            'module': classifier.state_dict(),
            'train_accuracy': train_instance_acc,
            'test_accuracy': test_instance_acc,
            'optimizer': optimizer.state_dict()
        }
        torch.save(save_dict, SAVE_PATH)
        print("::save successful!")'''
