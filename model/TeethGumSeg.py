import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.init as init
import torch.utils.data as data
from PointNetPartSeg import load_module_opti, feature_transform_reguliarzer, get_loss, get_model
from PointNet_plus import get_loss as get_loss_plus
from PointNet_plus import get_model as get_model_plus
from processing_data import convert_to_datasets
from torch import nn
from sklearn.metrics import accuracy_score
import os

# ----consts---- #
# mutable    when you want to assign value, using 'global'
lr = 1e-3
epochs = 200
batch_size = 8
part_num = 2
test_rate = 0.2
path = r'/home/Wangling/czh/teeth/teeth_raw'
read_path = r'/home/Wangling/czh/logs/pointnet_2class_weight.pth'
save_path = r'/home/Wangling/czh/logs/pointnet_2class_weight.pth'
log_path = r'/home/Wangling/czh/logs/pointnet_2class.log'

os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5,6'
# torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
# torch.distributed.init_process_group(backend="nccl")
# model = DistributedDataParallel(model)  # device_ids will include all GPU devices by default


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
        optim_params = optimizer.state_dict()
        schud_params = scheduler.state_dict()
        static_dicts = {
            'module_params': module_params,
            'optim_params': optim_params,
            'schud_params': schud_params,
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
    best_acc = 0.0
    logger = get_logger(log_path)
    module = get_model(part_num=part_num, normal_channel=False, device='cuda:2')
    # module = get_model_plus(part_num)
    #module, _ = load_module_opti(read_path, part_num)
    # print('load module successfully')
    module.to('cuda:2')
    # module = DistributedDataParallel(model)  # device_ids will include all GPU devices by default
    # initialize(module)
    # logger.info('module get initialized')
    loss = get_loss(device='cuda:2').cuda()
    # loss = loss_for_teeth().cuda(1)
    optimizer = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15, gamma=0.8)  # lr reduce 20% every 15 epochs

    datasets_train, datasets_test = convert_to_datasets(path=path, test_rate=test_rate, predict=False,
                                                       seg_classes=part_num, scale=True, is_raw=False)
    #datasets_train, datasets_test = convert_to_datasets_four_class(path=path, scale=True, test_size=test_rate)
    print(len(datasets_train[0][0]))
    logger.info('scale is True')
    dataloader_train = data.DataLoader(datasets_train, batch_size=batch_size, shuffle=True, drop_last=False)
    dataloader_test = data.DataLoader(datasets_test, batch_size=batch_size, shuffle=True, drop_last=False)
    # ---train--- #
    draw_loss_lst_train = []
    draw_acc_lst_train = []
    draw_acc_lst_test = []
    draw_loss_lst_test = []

    module.train()
    for ep in range(epochs):
        train_acc_lst = []  # put every train accuracy in this list and calculate it's mean at the end
        train_loss_lst = []
        for X, y in dataloader_train:
            X = X.cuda(2).transpose(1, 2).float()
            y = y.cuda(2).long()
            y = y.view(-1, 1)[:, 0]

            seg_pred, trans_pred = module(X, for_teeth=True)
            # seg_pred, trans_pred = module(X)
            seg_pred = seg_pred.contiguous().view(-1, part_num)  # [bs*num_pts, 5]
            pred_choice = seg_pred.data.max(1)[1]
            optimizer.zero_grad()
            l = loss(seg_pred, y, trans_pred)
            # l = loss(y, seg_pred, trans_pred)
            # l = loss(seg_pred, y)
            l.backward()
            train_acc = accuracy_score(pred_choice.detach().cpu().numpy(), y.view(-1).detach().cpu().numpy())
            # train_acc = compute_acc(y, pred_choice)
            train_acc_lst.append(train_acc)
            train_loss_lst.append(l.detach().cpu().numpy())
            optimizer.step()
            scheduler.step()
        draw_acc_lst_train.append(np.mean(train_acc_lst)), draw_loss_lst_train.append(np.mean(train_loss_lst))
        logger.info(
            'ep%d, accuracy is %.3f, loss is %.3f' % ((ep + 1), np.mean(train_acc_lst), np.mean(train_loss_lst)))
        # ---test--- #
        module.eval()
        with torch.no_grad():
            test_acc_lst = []
            test_loss_lst = []
            for X, y in dataloader_test:
                X = X.cuda(2).transpose(1, 2).float()
                y = y.cuda(2).long()
                y = y.view(-1, 1)[:, 0]

                seg_pred, trans_pred = module(X, for_teeth=True)
                # seg_pred, trans_pred = module(X)
                seg_pred = seg_pred.contiguous().view(-1, part_num)  # [bs*num_pts, 5]
                pred_choice = seg_pred.data.max(1)[1]
                test_acc = accuracy_score(pred_choice.detach().cpu().numpy(), y.view(-1).detach().cpu().numpy())
                l = loss(seg_pred, y, trans_pred)
                # l = loss(seg_pred, y)
                # test_acc = compute_acc(y, pred_choice)  # 去处背景点的影响
                test_acc_lst.append(test_acc)
                train_loss_lst.append(l.detach().cpu().numpy())
            draw_acc_lst_test.append(np.mean(test_acc_lst))
            draw_loss_lst_test.append(np.mean(test_loss_lst))
            logger.info('test accuracy is %.3f' % np.mean(test_acc_lst))
            if np.mean(test_acc_lst) > best_acc:
                best_acc = np.mean(test_acc_lst)
            break_flag = early_stop(module, save_path, np.mean(test_acc_lst), logger, ep, 0.832)
            if break_flag:
                break  # save module already

    # ---save module--- #
    if not break_flag:
        module_params = module.state_dict()
        optim_params = optimizer.state_dict()
        schud_params = scheduler.state_dict()

        static_dicts = {
            'module_params': module_params,
            'optim_params': optim_params,
            'schud_params': schud_params,
            'epochs': epochs,
            'train_ep': epochs,
            'best_acc': best_acc
        }
        torch.save(static_dicts, save_path)
        logger.info('save module')

    # ---draw accuracy and loss--- #
    time_train = range(len(draw_loss_lst_train))

    plt.figure()
    los_line = plt.plot(time_train, draw_loss_lst_train)
    acc_line = plt.plot(time_train, draw_acc_lst_train)
    plt.title("train accuracy and loss")
    plt.xlabel("epochs")
    plt.ylabel("loss/acc")
    plt.legend([los_line, acc_line], ['loss', 'accuracy'])
    plt.savefig('/home/Wangling/czh/loss/2_class_train.png')

    plt.figure()
    plt.plot(range(len(draw_acc_lst_test)), draw_acc_lst_test)
    plt.plot(range(len(draw_loss_lst_test)), draw_loss_lst_test)
    plt.xlabel("epochs")
    plt.ylabel("accuracy/loss")
    plt.savefig('/home/Wangling/czh/loss/2class_test.png')
    logger.info("save fig")
