import numpy as np
import torch
import torch.tensor as tensor
import torch.utils.data as D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from load_data import read_teeth_data, read_teeth_data_four_class, read_teeth_data_five_class
from utils.open3d_utils import array_to_pointcloud
from utils.utils import *
import os
from tqdm import tqdm

# StandardScaler, MaxAbsScaler, Normalizer, RobustScaler
norm = StandardScaler()  # norm function


def __label_data(lst, all, scale=False):
    """
    为每一个点标记label
    打5个标签
    :param lst: 输入的原输入 type: list
    :param scale: 是否进行归一化
    :return: 输出标签
    """
    global all_raw
    labels = []
    all_lst = all.tolist()
    for i in range(len(lst)):  # 第i颗牙齿
        name, raw = lst[i]
        raw_lst = raw.tolist()
        raw_len = len(raw)
        cn = 0
        for elem in raw_lst:
            if elem in all_lst:
                all_lst.remove(elem)
            else:
                cn += 1
        print(cn, ' points not in')
        label_lst = [name - 3] * raw_len  # [4,5,6,7]->[1,2,3,4]
        labels.append(label_lst)

        if i != 0:
            all_raw = np.concatenate([all_raw, raw], axis=0)
        else:
            all_raw = raw

    rst_len = len(all_lst)
    rst_label = [0] * rst_len
    labels.append(rst_label)

    all = np.concatenate([all_raw, np.array(all_lst)], axis=0)
    sub_len = []
    for i in labels:
        sub_len.append(len(i))
    labels = sum(labels, [])  # flatten
    if scale:
        all = norm.fit_transform(all)
    return labels, all, sub_len


def __label_data_two_class(lst, all, scale=True):
    """
    打2类标签
    :param lst:
    :param all:
    :return:
    """
    all_raw = None
    labels = []
    all_lst = all.tolist()
    for i in range(len(lst)):  # 第i颗牙齿
        name, raw = lst[i]
        raw_lst = raw.tolist()
        raw_len = len(raw)
        cn = 0
        for elem in raw_lst:
            if elem in all_lst:
                all_lst.remove(elem)
            else:
                cn += 1
        print(cn, ' points not in')
        label_lst = [1] * raw_len
        labels.append(label_lst)

        if i != 0:
            all_raw = np.concatenate([all_raw, raw], axis=0)
        else:
            all_raw = raw

    rst_len = len(all_lst)
    rst_label = [0] * rst_len
    labels.append(rst_label)

    all = np.concatenate([all_raw, np.array(all_lst)], axis=0)
    sub_len = []
    for i in labels:
        sub_len.append(len(i))
    labels = sum(labels, [])  # flatten
    if scale:
        all = norm.fit_transform(all)
    return labels, all, sub_len


def __check_input(all, sub_len):
    color_all = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [.2, .4, .8], [1, 1, .3], [.2, .7, 1], [.5, 1, 1], [.4, .5, .6],
                 [.6, .9, .2]]
    t = 0
    pcd_all = []
    for j in range(len(sub_len)):
        seg_teeth = all[t:t + sub_len[j]]
        color = color_all[j]
        pcd = array_to_pointcloud(seg_teeth)
        pcd.paint_uniform_color(color)
        pcd_all.append(pcd)
        t += sub_len[j]
    o3d.visualization.draw_geometries(pcd_all)


def remove_duplicate_pts_for_four_parts(read_path, save_path):
    dir_start_num = len(os.listdir(save_path)) + 1
    datasets = convert_to_datasets_four_class(read_path, test_size=None, scale=False, down_sample_pts=None)
    X_lst = datasets.X
    y_lst = datasets.y
    for X, y in zip(X_lst, y_lst):
        t_all = [X[y == 0], X[y == 1], X[y == 2], X[y == 3]]
        t_save = []

        for t in t_all:
            t_save.append(density_adaptive_downsampling(t, size=2600))
        # all_save = density_adaptive_downsampling(X_all)  # size=8192

        # save
        joined_path = os.path.join(save_path, str(dir_start_num))
        # make dir
        os.makedirs(joined_path)
        for i, save_ti in enumerate(t_save, start=4):
            temp_t_path = os.path.join(joined_path, '2' + str(i) + '.txt')
            np.savetxt(temp_t_path, save_ti, fmt="%.8f")
        # all_joined_path = os.path.join(joined_path, 'all.txt')
        # np.savetxt(all_joined_path, all_save, fmt='%.8f')
        dir_start_num += 1
    print("save successfully!")


def remove_duplicate_pts_and_save_to_txt_for_twopart(read_path, save_path, scale=False):
    """
    将重复点移除 训练时就不需要重复处理
    :param read_path: 需要处理txt文件的根目录
    :param save_path: 存储文件的根目录
    :param scale: 是否进行归一化处理
    :return: None
    """
    dir_start_num = len(os.listdir(save_path)) + 1
    datasets = convert_to_datasets(read_path, predict=True, scale=scale, is_raw=True)
    X_lst = datasets.X
    y_lst = datasets.y
    for X, y in zip(X_lst, y_lst):
        teeth_segged = X[y == 1]
        mask = y == 0
        bcg = X[mask]
        ##### 6-4 #####
        # TODO: need test
        bcg_downsp = density_adaptive_downsampling(bcg)
        teeth_segged_downsp = density_adaptive_downsampling(teeth_segged)
        teeth_segged, bcg = teeth_segged_downsp, bcg_downsp
        if teeth_segged.shape[0] + bcg.shape[0] < 8192:
            print("warning, points num less than 8192, may raise label error")
        #######
        os.makedirs(os.path.join(save_path, str(dir_start_num)))
        teeth_txt_path = os.path.join(save_path, str(dir_start_num), '1.txt')
        rst_txt_path = os.path.join(save_path, str(dir_start_num), 'rest.txt')
        np.savetxt(teeth_txt_path, teeth_segged, fmt='%.8f')
        np.savetxt(rst_txt_path, bcg, fmt='%.8f')
        dir_start_num += 1


class mydatasets(D.Dataset):
    def __init__(self, X, y, sample_pts=8192):
        super(mydatasets, self).__init__()
        self.X = []
        self.y = []
        if sample_pts is not None:
            self.__down_sample(X, y, sample_pts=sample_pts)
        else:
            for Xi, yi in zip(X, y):
                self.X.append(Xi)
                self.y.append(yi)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.y)

    def __down_sample(self, X, y, sample_pts):
        # 进行降采样保证每一个batch里面的点数都是相同的
        for resample_X, resample_y in zip(X, y):
            length = len(resample_X)
            perm_resample = torch.randperm(length)
            torch.manual_seed(2021)
            self.X.append(resample_X[perm_resample][:sample_pts])
            self.y.append(resample_y[perm_resample][:sample_pts])


# for 2 seg class
def convert_to_datasets_for_predict(path, decimals=3, scale=True):
    """
    输入为预测时的txt文件 包含所有点
    :param path: 文件所在的根目录
    path--
        |- teeth1.txt
        |- teeth2.txt
        |...
    :param decimals: 精度
    :return: nn.Datasets
    """
    all_pts_tensor = None
    all_cat = []
    DirLst = os.listdir(path)
    assert len(DirLst) >= 1

    for i in range(len(DirLst)):
        if i == len(DirLst) - 1 and (i - 1) % 2 == 1:
            NowPath = os.path.join(path, DirLst[0])  # every teeth path
            AllPts = np.loadtxt(NowPath)
            AllPts = np.round(AllPts, decimals=decimals)
            if scale:
                AllPts = norm.fit_transform(AllPts)
            AllPts = tensor(AllPts[torch.randperm(len(AllPts))][:8192, :][None, :, :])
            all_pts_tensor = torch.cat([AllPts, AllPts], dim=0)
            all_cat.append(all_pts_tensor)
        if i % 2 == 0:
            first_path = os.path.join(path, DirLst[0])
            all_pts = np.round(np.loadtxt(first_path), decimals=decimals)
            if scale:
                all_pts = norm.fit_transform(all_pts)
            all_pts_tensor = tensor(all_pts)
            all_pts_tensor = all_pts_tensor[torch.randperm(len(all_pts_tensor))][:8192, :][None, :, :]
        else:
            NowPath = os.path.join(path, DirLst[i])  # every teeth path
            AllPts = np.loadtxt(NowPath)
            AllPts = np.round(AllPts, decimals=decimals)
            if scale:
                AllPts = norm.fit_transform(AllPts)
            AllPts = tensor(AllPts)
            AllPts = AllPts[torch.randperm(len(AllPts))][:8192, :][None, :, :]
            all_pts_tensor = torch.cat([all_pts_tensor, AllPts], dim=0)
            all_cat.append(all_pts_tensor)

    return all_cat


# noinspection SpellCheckingInspection
def convert_to_datasets(path, test_rate=None, predict=False, seg_classes=2, scale=True, is_raw=True, is_splitic=False):
    """
    for torch Datasets input
    :param is_splitic: 数据是否划分为train和test
    :param is_raw: 是否是未经处理的数据
    :param scale: 是否归一化
    :param seg_classes: 分为几类 2或者是5
    :param predict: 预测时置为True 这里的预测是指训练时的验证集
    :param test_rate: 测试集样本数占所有样本的比例
    :param path: reading path of teeth
    :return: Datasets containing X,y(teeth data and labels of every point)
    """

    #### 6/4 ####
    def __down_sample_for_voxel_processing(teeth, label, size=8192):
        if isinstance(teeth, np.ndarray):
            teeth = tensor(teeth)
            label = tensor(label)
        length = len(teeth)
        perm_resample = torch.randperm(length)
        torch.manual_seed(2021)
        X = teeth[perm_resample][:size].numpy()
        y = label[perm_resample][:size].numpy()
        return X, y

    def __make_datasets(iter_data, all_d, scale):
        all_data = []
        all_labels = []
        for res, all in zip(iter_data, all_d):
            y, data, sub_len = label_data(res, all, scale=scale)  # y(label) data(teeth data) sub_len(list)
            all_labels.append(tensor(y))
            all_data.append(tensor(data))
        datasets = mydatasets(all_data, all_labels, sample_pts=None)
        return datasets

    def __make_datasets_for_is_not_raw(iter_data, all_d, scale):
        all_data = []
        all_raw_data = []
        all_labels = []
        for teeth, rst in zip(iter_data, all_d):
            labels = []
            teeth = teeth[0][1]
            teeth_len = len(teeth)
            rst_len = len(rst)
            labels.append([1] * teeth_len)
            labels.append([0] * rst_len)
            all_pts = np.concatenate([teeth, rst], axis=0)
            all_raw_data.append(all_pts)  # raw data
            if scale:
                all_pts = norm.fit_transform(all_pts)
            all_data.append(all_pts)
            all_labels.append(sum(labels, []))

        #### 6/4 ####
        teeth_data = []
        label_data = []
        teeth_data_raw = []
        for t, tr, l in zip(all_data, all_raw_data, all_labels):
            all_pts, labels = __down_sample_for_voxel_processing(t, l)
            all_pts_raw, _ = __down_sample_for_voxel_processing(tr, l)
            teeth_data.append(all_pts)
            label_data.append(labels)
            teeth_data_raw.append(all_pts_raw)
        all_data, all_raw_data, all_labels = teeth_data, teeth_data_raw, label_data
        ########

        datasets = mydatasets(tensor(all_data), tensor(all_labels))
        RawDatasets = mydatasets(tensor(all_raw_data), tensor(all_labels))
        return datasets, RawDatasets

    try:  # value check
        if seg_classes != 2 and seg_classes != 5:
            raise ValueError
    except ValueError:
        print("seg_class must be set 2 or 5")

    if is_splitic:
        predict = True

    if seg_classes == 2:
        label_data = __label_data_two_class
    else:
        label_data = __label_data  # 5 class

    if is_raw:
        if not predict:
            iter_train, iter_test, all_train, all_test = read_teeth_data(path, test_rate=test_rate, predict=False)
            datasets_train = __make_datasets(iter_train, all_train, scale=scale)
            datasets_test = __make_datasets(iter_test, all_test, scale=scale)
            print('processing success')
            return datasets_train, datasets_test
        else:  # predict # remove_duplicate_pts_and_to_txt_for_twopart
            ### warning 只被remove_duplicate_pts_and_to_txt_for_twopart函数调用
            # all_labels_pre = []
            # all_data_pre = []
            iter_pre, all_pts = read_teeth_data(path, predict=True)
            datasets_pre = __make_datasets(iter_pre, all_pts, scale=scale)
            print('processing success')
            return datasets_pre
    else:  # 经过remove和down_sample处理 scale?
        if not predict:
            iter_train, iter_test, rst_train, rst_test = read_teeth_data(path, test_rate=test_rate, predict=False,
                                                                         is_raw=False)
            datasets_train, _ = __make_datasets_for_is_not_raw(iter_train, rst_train, scale=scale)
            datasets_test, _ = __make_datasets_for_is_not_raw(iter_test, rst_test, scale=scale)
            print('processing success')
            return datasets_train, datasets_test
        else:  # predict
            iter_pre, rst_pts = read_teeth_data(path, predict=True, is_raw=False)
            datasets_pre, RawDatasets = __make_datasets_for_is_not_raw(iter_pre, rst_pts, scale=scale)
            print('processing success')
            return datasets_pre, RawDatasets


# pct or pointnet
def convert_to_datasets_four_class(path, test_size=None, scale=True, decimals=3, down_sample_pts=8192):
    # TODO 在打标签之前就进行下采样
    teeth_lst, _ = read_teeth_data_four_class(path, decimals=decimals)
    all_label = []
    all_teeth = []

    for p in teeth_lst:
        label_f, seg_teeth_f = p[0]
        label_lst = [[label_f] * len(seg_teeth_f)]
        for i in range(1, len(p)):
            label, seg_teeth = p[i][0], p[i][1]
            label_lst.append([label] * len(seg_teeth))
            seg_teeth_f = np.concatenate([seg_teeth_f, seg_teeth], axis=0)
        label_lst = tensor(sum(label_lst, []))
        if scale:
            seg_teeth_f = norm.fit_transform(seg_teeth_f)
        seg_teeth_f = tensor(seg_teeth_f)
        all_label.append(label_lst)
        all_teeth.append(seg_teeth_f)
    if test_size is not None:
        X_train, X_test, y_train, y_test = train_test_split(all_teeth, all_label, test_size=test_size)
        return mydatasets(X_train, y_train), mydatasets(X_test, y_test)
    else:  # test_size is None
        datasets = mydatasets(all_teeth, all_label, sample_pts=down_sample_pts)
        return datasets


# show four seg teeth
def show_four_seg_teeth(data_loader: D.DataLoader):
    for X, y in data_loader:
        for Xi, yi in zip(X, y):
            Xi = Xi.squeeze()
            X1 = Xi[yi == 1].numpy()
            X2 = Xi[yi == 2].numpy()
            X3 = Xi[yi == 3].numpy()
            X4 = Xi[yi == 4].numpy()
            pcd1 = array_to_pointcloud(X1).paint_uniform_color([0, 0, 1])
            pcd2 = array_to_pointcloud(X2).paint_uniform_color([0, 1, 0])
            pcd3 = array_to_pointcloud(X3).paint_uniform_color([1, 0, 0])
            pcd4 = array_to_pointcloud(X4).paint_uniform_color([.5, .5, .5])

            o3d.visualization.draw_geometries([pcd1, pcd2, pcd3, pcd4])


# 6/27
def make_bg_for_5class_seg(read_path, save_path):
    teeth_lst, all_lst = read_teeth_data_four_class(read_path)
    exist_dir_num = len(os.listdir(save_path))
    save_dir_num = exist_dir_num
    for one2four_t, _all in tqdm(zip(teeth_lst, all_lst), total=len(teeth_lst)):
        save_dir_num += 1
        ins_save_path = os.path.join(save_path, str(save_dir_num))
        os.makedirs(ins_save_path)
        t = one2four_t[0][1]
        ds_t = density_adaptive_downsampling(t, size=2600)
        np.savetxt(os.path.join(ins_save_path, '1.txt'), ds_t)
        for i in range(1, 4):
            ins_t = one2four_t[i][1]
            ins_ds_t = density_adaptive_downsampling(ins_t, size=2600)
            np.savetxt(os.path.join(ins_save_path, str(i + 1) + '.txt'), ins_ds_t)
            t = np.concatenate((t, ins_t))
        bg = []
        for alli in _all:
            if alli not in t:
                bg.append(alli)
        bg = np.array(bg)
        ds_bg = density_adaptive_downsampling(bg, size=2600 * 2)
        np.savetxt(os.path.join(ins_save_path, 'bg.txt'), ds_bg)


# 6/27
def convert_to_datasets_five_class(path, test_size=None, scale=True, decimals=3, down_sample_pts=8192):
    teeth_lst, bg_lst = read_teeth_data_five_class(path, decimals=decimals)
    all_label = []
    all_teeth = []
    for p, bg in zip(teeth_lst, bg_lst):
        # 1->4
        label_f, seg_teeth_f = p[0]
        label_lst = [[label_f] * len(seg_teeth_f)]

        for i in range(1, len(p)):
            label, seg_teeth = p[i][0], p[i][1]
            label_lst.append([label] * len(seg_teeth))
            seg_teeth_f = np.concatenate([seg_teeth_f, seg_teeth], axis=0)
        # bg:0
        label_lst.append([0] * len(bg))
        seg_teeth_f = np.concatenate([seg_teeth_f, bg], axis=0)

        # scale
        if scale:
            seg_teeth_f = norm.fit_transform(seg_teeth_f)

        # convert to tensor
        label_lst = tensor(sum(label_lst, []))
        seg_teeth_f = tensor(seg_teeth_f)
        # append to all_label and all_teeth
        all_label.append(label_lst)
        all_teeth.append(seg_teeth_f)

    # split to test and train
    if test_size is not None:
        X_train, X_test, y_train, y_test = train_test_split(all_teeth, all_label, test_size=test_size)
        return mydatasets(X_train, y_train), mydatasets(X_test, y_test)
    else:  # test_size is None
        datasets = mydatasets(all_teeth, all_label, sample_pts=down_sample_pts)
        return datasets


if __name__ == '__main__':
    import open3d as o3d

    datasets = convert_to_datasets_five_class('D:/desktop/make_bg', scale=True)
    dl = D.DataLoader(datasets, batch_size=1)
    for X, y in dl:
        y = y[0]
        X = X[0]

        X_pcd = array_to_pointcloud(X[y == 0])
        o3d.visualization.draw_geometries([X_pcd])

    # make_bg_for_5class_seg(r'D:\desktop\make_bg_read', r'D:\desktop\make_bg')
    # datasets, _ = convert_to_datasets('D:/desktop/2', scale=True, is_raw=False, predict=True)
    # dataloader = D.DataLoader(datasets, batch_size=2)
    # for X, y in dataloader:
    #     for Xi, yi in zip(X, y):
    #         Xi = Xi.squeeze()
    #         X1 = Xi[yi == 1].numpy()
    #         X2 = Xi[yi == 2].numpy()
    #         X3 = Xi[yi == 3].numpy()
    #         X4 = Xi[yi == 4].numpy()
    #         pcd1 = array_to_pointcloud(X1).paint_uniform_color([0, 0, 1])
    #         pcd2 = array_to_pointcloud(X2).paint_uniform_color([0, 1, 0])
    #         pcd3 = array_to_pointcloud(X3).paint_uniform_color([1, 0, 0])
    #         pcd4 = array_to_pointcloud(X4).paint_uniform_color([.5, .5, .5])
    #
    #         o3d.visualization.draw_geometries([pcd1, pcd2, pcd3, pcd4])

    # datasets = convert_to_datasets_for_predict(r'D:/desktop/2',scale=True)
    #
    # module = get_model(part_num=2, normal_channel=False, device='cuda:0').cuda()
    # module, _ = load_module_opti('D:/codeblock/code/dachuang/logs/pretrained_module.pth')
    # dataloader = D.DataLoader(datasets, batch_size=2)
    # # module, _ = load_module_opti('D:/codeblock/code/dachuang/logs/pointnet_teeth_program_read.pth')
    # module.cuda()
    # for X, y in dataloader:
    #     X = X.cuda().transpose(2, 1).float()
    #     seg_pred, trans_pred = module(X, for_teeth=True)
    #     pred_choice = seg_pred.data.max(2)[1]
    #     X = X.transpose(2, 1)
    #     for i, j in zip(X, pred_choice):
    #         module_predict_visualization(i, j)
    # datasets, RawDatasets = convert_to_datasets('D:/desktop/2', predict=True, is_raw=False, scale=True)
    # dataloader = D.DataLoader(datasets, batch_size=2)
    # RawDataloader = D.DataLoader(RawDatasets, batch_size=2)
    # module.eval()
    # for Xy, Xy1 in zip(dataloader, RawDataloader):
    #     X, y = Xy
    #     X1, y1 = Xy1
    #     X = X.cuda().transpose(2, 1).float()
    #     y = y.cuda().long()
    #     seg_pred, trans_pred = module(X, for_teeth=True)
    #     pred_choice = seg_pred.data.max(2)[1]
    #     acc = accuracy_score(y.view(-1).cpu().detach().numpy(), pred_choice.view(-1).detach().cpu().numpy())
    #     print(acc)
    #     X = X.transpose(2, 1)
    #     for i, j in zip(X1, pred_choice):
    #         module_predict_visualization(i, j)
    # pcd = array_to_pointcloud(X[0])
    # X1 = X[0][y[0] == 1].numpy()
    # X1_rst = X[0][y[0] == 0].numpy()
    # X1 = array_to_pointcloud(X1).paint_uniform_color([0, 0, 1])
    # X1_rst = array_to_pointcloud(X1_rst).paint_uniform_color([0, 1, 0])
    # o3d.visualization.draw_geometries([X1, X1_rst])
    # remove_duplicate_pts_for_four_parts(r'D:\desktop\teeth_program_read\txt格式（测试001-010）',
    #                                     r'D:/desktop/save')
    # remove_duplicate_pts_and_save_to_txt_for_twopart(r"D:\desktop\1",
    #                                                  r"D:\desktop\4")
    # module_path = r"D:\codeblock\code\dachuang\logs\pointnet_teeth_2class.pth"
    # module, _ = load_module_opti(module_path, part_num=2)
    # module.to('cuda:0')
    # datasets = convert_to_datasets(path=r'D:/desktop/1', predict=True, scale=True)
    # datasets2 = convert_to_datasets(path=r'D:/desktop/1', predict=True, scale=False)
    # dataloader = D.DataLoader(datasets, batch_size=2)
    # dataloader2 = D.DataLoader(datasets2, batch_size=2)
    #
    # for [X, y], [X2, _] in zip(dataloader, dataloader2):
    #     print(X)
    #     X = X.cuda().transpose(1, 2).float()
    #     y = y.cuda().long()
    #     seg_pred, _ = module(X, for_teeth=True)
    #     pred_choice = seg_pred.data.max(2)[1]
    #     X = X.transpose(1, 2)
    #     for i, j in zip(X2, pred_choice):
    #         module_predict_visualization(i, j)
