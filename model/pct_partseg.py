from torch import cat as concat
import torch.utils.data as data
from processing_data import convert_to_datasets_four_class, convert_to_datasets_five_class
# from prac import get_logger, early_stop
from torch import nn
from sklearn.metrics import accuracy_score
import time
from utils.open3d_utils import *

# ----time----#
now_time = time.strftime('%Y-%m-%d-%H', time.localtime())  # eg. '2021-05-06-11'
# ----consts---- #
# mutable    when you want to assign value, using 'global'
lr = 1e-3
epochs = 50
total_epochs = 50
batch_size = 1
part_num = 5
# test_rate = 0.2
# path = r'D:\desktop\teeth_program_read\rotate（1-40）'
path = r'D:\desktop\test_5class'
read_path = r'D:\codeblock\code\dachuang\logs\PCT5_weight_2021-07-01-01.pth'
save_path = r'/home/Wangling/czh/logs/PCT_weight_' + now_time + '.pth'
log_path = r'/home/Wangling/czh/logs/PCT_logs_' + now_time + '.log'
break_flag = False


# os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3,4,5,6,7'


class Point_Transformer_partseg(nn.Module):
    def __init__(self, part_num=50, inplane=3):
        super(Point_Transformer_partseg, self).__init__()
        self.part_num = part_num
        self.conv1 = nn.Conv1d(inplane, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

        self.bn1 = nn.GroupNorm(4, 128)
        self.bn2 = nn.GroupNorm(4, 128)

        self.sa1 = SA_Layer(128)
        self.sa2 = SA_Layer(128)
        # self.sa3 = SA_Layer(128)
        # self.sa4 = SA_Layer(128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(256, 1024, kernel_size=1, bias=False),
                                       nn.GroupNorm(4, 1024),
                                       nn.LeakyReLU())

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                        nn.GroupNorm(4, 64),
                                        nn.LeakyReLU())

        self.convs1 = nn.Conv1d(1024 * 3, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, self.part_num, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU6()

        self.log_sfm = nn.LogSoftmax(1)

        self.apply(self.weight_init)

    def weight_init(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def _forward_imple(self, x):
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        # x.register_hook(save_grad('first_layer_x'))
        x = self.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        # x3 = self.sa3(x2)
        # x4 = self.sa4(x3)
        x = concat((x1, x2), dim=1)
        x = self.conv_fuse(x)
        x_max = x.max(2, keepdim=True)[0]
        x_avg = torch.mean(x, 2)
        x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        # cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        # cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)
        x_global_feature = concat((x_max_feature, x_avg_feature), 1)  # 1024
        x = concat((x, x_global_feature), 1)  # 1024 * 3
        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)
        x = self.log_sfm(x)
        return x

    def forward(self, x):
        return self._forward_imple(x)


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        # self.q_conv.conv.weight = self.k_conv.conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(128)
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

if __name__ == "__main__":
    # net = Point_Transformer_partseg()
    # a = torch.randn(10, 3, 8192)
    # print(net(a).shape)
    module = Point_Transformer_partseg(part_num=part_num).cuda()

    model_dict = torch.load(read_path, map_location='cpu')['module_params']

    # model_dict_clone = model_dict.copy()
    # for key, value in model_dict_clone.items():
    #     if key.endswith(('running_mean', 'running_var')):
    #         del model_dict[key]

    module.load_state_dict(model_dict, False)
    module.cpu()
    print('module load success')

    datasets = convert_to_datasets_five_class(path=path, scale=True)
    datasets2 = convert_to_datasets_five_class(path=path, scale=False)
    dataloader = data.DataLoader(datasets, batch_size=batch_size, shuffle=False, drop_last=False)
    dataloader2 = data.DataLoader(datasets2, batch_size=batch_size, shuffle=False, drop_last=False)

    module.eval()
    for Xy, Xy2 in zip(dataloader, dataloader2):
        X, y = Xy
        X2, y2 = Xy2
        X = X.transpose(1, 2).float()
        y = y.long()
        y = y.view(-1, 1)[:, 0]
        seg_pred = module(X).transpose(1, 2)
        seg_pred = seg_pred.contiguous().view(-1, part_num)  # [bs*num_pts, 5]
        # print(seg_pred.shape)
        pred_choice = seg_pred.data.max(1)[1]
        test_acc = accuracy_score(pred_choice.detach().cpu().numpy(), y.view(-1).detach().cpu().numpy())
        print(test_acc)
        # X = X.transpose(1, 2)
        # pred_choice = pred_choice.contiguous().view(batch_size, -1)
        # for i, j in zip(X2, pred_choice):
        #     bounding_box_visualization(i, j)
        X = X.transpose(1, 2).cpu().squeeze().numpy()
        pred_choice = pred_choice.cpu().numpy()
        color_all = [[.5, .5, .5], [0, 0, 1], [0, 1, 0], [1, 0, 0], [.6, .1, .8], [1, 1, .3],
                     [.2, .7, 1], [.5, 1, 1], [.4, .5, .6]]

        bg = array_to_pointcloud(X[np.argwhere(pred_choice == 0)].squeeze())  # background
        t1 = array_to_pointcloud(X[np.argwhere(pred_choice == 1)].squeeze())
        t2 = array_to_pointcloud(X[np.argwhere(pred_choice == 2)].squeeze())
        t3 = array_to_pointcloud(X[np.argwhere(pred_choice == 3)].squeeze())
        t4 = array_to_pointcloud(X[np.argwhere(pred_choice == 4)].squeeze())
        # uniform color
        t2.paint_uniform_color(color_all[2])
        t3.paint_uniform_color(color_all[3])
        t4.paint_uniform_color(color_all[4])
        bg.paint_uniform_color(color_all[0])
        t1.paint_uniform_color(color_all[1])
        o3d.visualization.draw_geometries([bg, t1, t2, t3, t4])
