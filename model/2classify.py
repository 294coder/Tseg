import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.init as init
import torch.utils.data as data
from PointNetPartSeg import load_module_opti, feature_transform_reguliarzer, get_loss, get_model
from PointNet_plus import get_loss as get_loss_plus
from PointNet_plus import get_model as get_model_plus
from pct_partseg import Point_Transformer_partseg
from processing_data import convert_to_datasets, convert_to_datasets_four_class
from torch import nn
from sklearn.metrics import accuracy_score
import os

# ----consts---- #
# mutable    when you want to assign value, using 'global'
lr = 1e-3
epochs = 50
total_epochs = 50
batch_size = 1
part_num = 2
test_rate = 0.5
device = 'cpu'
path = r'D:\desktop\teeth_down_sample_voxel'
read_path = r'D:\codeblock\code\dachuang\logs\pct_2class6-test.pth'
np.interp


def get_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    fh = logging.FileHandler(log_path, mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def compute_acc(raw_y, raw_pred):
    # 去除背景点对准确度的影响 只关注四颗牙是否分割对
    cnt = 0
    nonzero_num = torch.count_nonzero(raw_y).item()
    # print(nonzero_num)
    for a, b in zip(raw_y, raw_pred):
        if a == b and a != 0:
            cnt += 1
    return cnt / nonzero_num


def initialize(net: nn.Module):
    with torch.no_grad():
        for name, param in net.named_parameters():
            if 'weight' in name:
                init.normal_(param, mean=0.0, std=0.5)
            if 'bias' in name:
                init.constant_(param, 0.0)


class loss_for_teeth(nn.Module):
    # noinspection SpellCheckingInspection
    def __init__(self, mat_diff_loss_scale=0.001):
        super(loss_for_teeth, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale
        self.nllloss = nn.NLLLoss()

    def forward(self, raw_y, raw_pred, trans_feats):
        nonzero_num = torch.count_nonzero(raw_y).item()
        length = len(raw_pred)
        l1 = self.nllloss(raw_pred, raw_y) * (nonzero_num / length) * 2
        l2 = feature_transform_reguliarzer(trans_feats) * self.mat_diff_loss_scale
        return l1 + l2


def early_stop(module, save_path, acc, logger, ep, shred=0.75):
    break_flag = False
    if acc > shred:
        # ---save module--- #
        module_params = module.state_dict()
        static_dicts = {
            'module_params': module_params,
            'epochs': epochs,
            'train_ep': ep,
            'best_acc': acc
        }
        torch.save(static_dicts, save_path)
        logger.info("early stop")
        logger.info("save module")
        break_flag = True
    return break_flag


if __name__ == "__main__":
    from pct_partseg import Point_Transformer_partseg
    from utils.open3d_utils import *

    module = Point_Transformer_partseg(part_num=part_num)
    # module = get_model_plus(part_num)
    module.to(device)
    module.load_state_dict(torch.load(read_path, map_location='cpu')['module_params'])
    print('load module successfully')

    loss = nn.NLLLoss()

    datasets1, _ = convert_to_datasets(path=path, scale=True, is_raw=False, test_rate=test_rate)
    # datasets2 = convert_to_datasets(path=path, scale=False, is_raw=False, test_rate=0.2)

    dataloader1 = data.DataLoader(datasets1, batch_size=batch_size, shuffle=False, drop_last=False)
    # dataloader2 = data.DataLoader(datasets2, batch_size=batch_size, shuffle=False, drop_last=False)
    module.eval()
    for X, y in dataloader1:

        X = X.transpose(1, 2).float().to(device)
        y = y.long().to(device)
        unique_y = torch.unique(y)
        y = y.view(-1, 1)[:, 0]
        seg_pred = module(X).transpose(1, 2)
        seg_pred = seg_pred.contiguous().view(-1, part_num)  # [bs*num_pts, 5]

        pred_choice = seg_pred.data.max(1)[1]
        test_acc = accuracy_score(pred_choice.detach().cpu().numpy(), y.view(-1).detach().cpu().numpy())
        print(test_acc)
        pred_choice = pred_choice.contiguous().view(batch_size, -1)
        for i, j in zip(X, pred_choice):
            i = i.transpose(0, 1)
            bounding_box_visualization(i, j, for_4class=False)
