import copy
import os

import numpy as np
import open3d as o3d
import pandas as pd
import torch as t
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from Three_registration import process_icp_usual
from get_division_polylines import points_index
from utils.open3d_utils import array_to_pointcloud

model_net_10_main_path = r"E:/ModelNet10/ModelNet10/ModelNet10/"
# model_net_10_main_path = r'/home/Wangling/czh/ModelNet10/ModelNet10/ModelNet10/'  # remote path
model_net_40_main_path = r"E:/modelnet40_normal_resampled/"


# model_net_40_main_path = r'/home/Wangling/czh/modelnet40_normal_resampled/'


# 这里会返回列表类型
def load_data_txt(path):
    file = open(path, 'r')
    data = file.read().split('\n')
    lst = _data_trans(data)
    return lst


def _data_trans(data):
    lst = []
    for num in data:
        num_list = num.split()
        lst.append([eval(i) for i in num_list])
    lst.pop()
    return lst


def load_data_csv(path):
    data = pd.read_csv(path, header=None)
    data = data.to_numpy().squeeze()
    lst = _data_trans(data)
    return lst


def stl_to_ply(path):
    source_mesh = o3d.io.read_triangle_mesh(path)
    o3d.io.write_triangle_mesh(r"D:\desktop\source_mesh_raw_data.ply", source_mesh)  # 将stl格式转换为ply格式
    source_ply = o3d.io.read_point_cloud(r"D:\desktop\source_mesh_raw_data.ply")
    return source_ply


def data_cat(data: list, axis=0):
    data1 = data[0]
    for i in range(1, len(data)):
        data1 = np.concatenate([data1, data[i]], axis=axis)
    return data1


# 统计当前文件夹的文件个数
def cout_file_num(path):
    count = 0
    for _, _, files in os.walk(path):
        for each in files:
            count += 1
    print('::find %d file(s)' % count)
    return count


class get_modelnet40_data():
    def __init__(self, batch_size=24, path=model_net_40_main_path, train=True):
        self.path = path
        self.batch_size = batch_size
        # self.choose_labels = choosed_labels  # for test
        # self.constrain_labels = constrain_labels  # for train
        self.all_labels = np.array(['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
                                    'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
                                    'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
                                    'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
                                    'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
                                    'wardrobe', 'xbox'])
        if train:
            self.f_train = open(self.path + "/modelnet40_train.txt")
            self.names = self.f_train.readlines()
            print("train num is %d" % len(self.names))
        else:
            self.f_test = open(self.path + "/modelnet40_test.txt")
            self.names = self.f_test.readlines()
            print("test num is %d" % len(self.names))

    def pcd(self):
        # 随机读取batch_size个文件
        if len(self.names) >= self.batch_size:
            lines = random.sample(self.names, self.batch_size)
            for l in lines:
                self.names.remove(l)
                l = l.strip('\n')
                label_lst = l.split('_')
                if len(label_lst) == 3:
                    label = label_lst[0] + '_' + label_lst[1]
                else:
                    label = label_lst[0]
                idx = np.where(self.all_labels == label)[0][0]
                batch_path = self.path + label + '/' + l + '.txt'
                all_coordiantes = read_off(batch_path)
                return idx, all_coordiantes

    def __iter__(self):
        return self

    def __next__(self):
        all_coordinates = []
        all_idx = []
        for _ in range(self.batch_size):
            if self.pcd() is not None:
                label_idx, coordinates = self.pcd()
                all_coordinates.append(coordinates)
                all_idx.append(label_idx)
        return all_idx, all_coordinates


def get_modelnet_data(read_file_num, train=True):
    """
    得到modelnet中的点的坐标
    如果train为False 则需要指定choosed_labels 表示测试模式
    考虑到显存限制 可能指定的class_num<10 所以测试的时候需要指定choosed_labels表示之前训练时的labels
    :param train True为训练模式  False为测试模式
    :param read_file_num: 每一个类别读进来的文件的个数
    :param class_num: 类别个数
    :return:返回labels 每个label的数字代码 点的坐标
    """
    global path1
    all_idx = []
    getted_labels = []
    all_coordinates = []

    # model_net_40
    main_path = model_net_40_main_path
    labels = np.array(['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
                       'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
                       'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
                       'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
                       'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
                       'wardrobe', 'xbox'])
    # print(labels)

    if train:
        f_train = open(main_path + "/modelnet40_train.txt")
        names = f_train.readlines()
        print("train num is %d" % len(names))

        lines = random.sample(names, read_file_num)
        for l in tqdm(lines):
            names.remove(l)
            l = l.strip('\n')
            label_lst = l.split('_')
            if len(label_lst) == 3:
                label = label_lst[0] + '_' + label_lst[1]
                getted_labels.append(label)
            else:
                label = label_lst[0]
                getted_labels.append(label)

            idx = np.where(labels == label)[0][0]
            batch_path = main_path + label + '/' + l + '.txt'
            coordiantes = read_off(batch_path)

            all_idx.append(idx)
            all_coordinates.append(coordiantes)

    else:  # test
        f_test = open(main_path + "/modelnet40_test.txt")
        names = f_test.readlines()
        print("train num is %d" % len(names))

        lines = random.sample(names, read_file_num)
        for l in tqdm(lines):
            names.remove(l)
            l = l.strip('\n')
            label_lst = l.split('_')
            if len(label_lst) == 3:
                label = label_lst[0] + '_' + label_lst[1]
                getted_labels.append(label)
            else:
                label = label_lst[0]
                getted_labels.append(label)

            idx = np.where(labels == label)[0][0]
            batch_path = main_path + label + '/' + l + '.txt'
            coordiantes = read_off(batch_path)

            all_idx.append(idx)
            all_coordinates.append(coordiantes)
    return all_idx, getted_labels, all_coordinates


def get_noted_teeth_data(noted_file_num: int, fusion_to_class=False, fusion_to_array=False):
    def _fusion_to_tensor(data_pcd, data_line, num_pts=7500):
        # 7500是data.Dataloader要求同一个batch里的数据个数必须相同
        l = []
        labels = []
        for i in range(len(data_pcd)):
            # pcd_length = len(data_pcd[i])
            line_length = len(data_line[i])

            # 降采样
            pcd = data_pcd[i]
            np.random.shuffle(pcd)
            pcd_num = num_pts - line_length
            pcd = pcd[:pcd_num]  # shuffle之后选择前pcd_num个点 实际画图之后可以达到效果

            pcd = t.from_numpy(pcd)
            line = t.tensor(data_line[i])

            all_points = t.cat((pcd, line), dim=0).float()
            all_points.requires_grad = True

            label = [0] * pcd_num
            temp = [1] * line_length
            for elem in temp:
                label.append(elem)
            l.append(all_points)
            labels.append(t.tensor(label, dtype=t.int64))

        return l, labels

    def _fusion_to_class(data_pcd, data_line):
        # all_data = t.cat((data_pcd, data_line), dim=0)
        class_list = []
        cls = []
        assert len(data_pcd) == len(data_line)
        for i in range(len(data_pcd)):
            for j in range(len(data_pcd[i])):
                point = points_index(j)
                point.coordinate = data_pcd[i][j]
                cls.append(point)
            for k in range(len(data_line)):
                point = points_index(k + len(data_pcd[i]))
                point.coordinate = data_line[i][k]
                point.cls = True
                cls.append(point)
            cls1 = copy.deepcopy(cls)
            class_list.append(cls1)
            cls.clear()
        return class_list

    line2 = []  # 总的线上点坐标 保留到小数点后2位
    data_pcd2 = []  # 总的点云的点坐标 保留到小数点后2位
    line = []  # 总的线上点坐标
    data_pcd = []  # 总的点云的点坐标
    for k in range(1, noted_file_num + 1):

        lines_main_path = r'D:/desktop/teeth_6_note/txt/polylines/' + str(k) + '/'

        file_num = cout_file_num(lines_main_path)
        assert file_num > 0
        # 第一条线上点的坐标
        lines_path = lines_main_path + str(k) + '_000000.txt'
        data_lines = load_data_txt(lines_path)
        # 得到线上点的坐标
        if file_num > 1:
            for i in range(1, file_num):
                lines_path = lines_main_path + str(k) + '_00000' + str(i) + '.txt'
                data1 = load_data_txt(lines_path)
                data_lines = data_cat([data_lines, data1])
        line.append(np.array(data_lines))
        # line2.append(np.around(np.array(data_lines), precision))  # 规范到小数点后2位 # 精度太高 不好找点
        # 得到点云坐标
        pcd_main_path = r'D:/desktop/teeth_6_note/txt/PointCloud/'
        pcd_path = pcd_main_path + 'Cloud' + str(k) + '.txt'
        data_pcd.append(np.array(load_data_txt(pcd_path)))
        # data_pcd2.append(np.around(np.array(load_data_txt(pcd_path)), precision))
    if fusion_to_class:
        list_class = _fusion_to_class(data_pcd, line)
        return list_class
    elif fusion_to_array:
        return data_pcd, line
    else:
        pts, labels = _fusion_to_tensor(data_pcd, line)
        return pts, labels


def _write_to_txt(array: np.ndarray, path: str):
    with open(path, 'a+') as f:
        for elem1 in array.squeeze():
            f.write(str(elem1).strip('[').strip(']') + '\n')
    f.close()
    print('::写入成功')


# 为PointNet做准备工作 根据论文 应该是不用配准的
def to_icp():
    matrix_lst = []
    target_path = r'D:/desktop/teeth_6_note/ply/1.ply'
    txt_path = r'D:/desktop/teeth_6_note/txt1/'
    _, line = get_noted_teeth_data(3, fusion_to_array=True)
    for i in range(2, 4):
        source_path = r'D:/desktop/teeth_6_note/ply/' + str(i) + '.ply'
        matrix = process_icp_usual(source_path, target_path, False)
        matrix_lst.append(matrix)
        pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud(source_path)
        pcd.transform(matrix)
        line_trans = array_to_pointcloud(line[i - 1])
        line_trans.transform(matrix)
        pcd_points = np.asarray(pcd.points)
        line_trans_points = np.asarray(line_trans.points)
        _write_to_txt(pcd_points, txt_path + 'PointCloud/Cloud' + str(i) + '.txt')
        _write_to_txt(line_trans_points, txt_path + 'polylines/' + str(i) + '/' + str(i) + '_000000.txt')


def read_off(path, model_net_40=True):
    f = open(path, encoding='utf-8')

    if not model_net_40:
        txt = f.readlines()
        num_points = eval(txt[1].split()[0])
        pcd = txt[2:2 + num_points]
        res = []
        for elem in pcd:
            # res = t.cat([res, t.tensor([eval(i) for i in elem.split()])], dim=1)
            res.append([eval(i) for i in elem.split()])
        # res = t.tensor(res)
        res = np.asarray(res)
        res = t.from_numpy(res)
        return res
    else:
        pcd = f.readlines()
        all = []
        for i in pcd:
            elem = []
            i = i[:-1]
            j = i.split(',')
            for k in range(3):
                elem.append(eval(j[k]))
            all.append(elem)
        all = np.array(all)
        return t.from_numpy(all)


def get_shapenet_data_path(
        train, train_test_size,
        base_path=r'D:\codeblock\code\dachuang\shapenetcore_partanno_segmentation_benchmark_v0_normal',
):
    category_path = os.path.join(base_path, 'synsetoffset2category.txt')

    def get_category() -> dict:
        """
        get label_name and encode_num
        :return: a dict with label_names as keys and encode_num is value. For example: {'Airplane': 0269115}
        """
        f_category = open(category_path)
        category = f_category.readlines()
        cate_to_dict = {}
        for i in range(len(category)):
            st = category[i].split('\t')
            cate_to_dict[st[0]] = st[1].strip('\n')
        return cate_to_dict

    def get_data_path(cate_dict) -> (list, list):
        labels_lst = list(cate_dict.values())
        all_dir_path = []
        for encode in labels_lst:
            path = os.path.join(base_path, encode)
            all_dir_path.append(path)
        data_txt_name = []
        for elem in all_dir_path:
            data_path = os.listdir(elem)
            data_txt_name.append(data_path)
        ret_labels_lst = []
        ret_data_txt_path = []
        for i in range(len(labels_lst)):
            length = len(data_txt_name[i])
            for j in range(length):
                ret_labels_lst.append(labels_lst[i])
                ret_data_txt_path.append(data_txt_name[i][j])
        return ret_data_txt_path, ret_labels_lst

    d = get_category()
    p, l = get_data_path(d)
    X_train, X_test, y_train, y_test = train_test_split(p, l, test_size=train_test_size)
    if train:
        return X_train, y_train, d
    else:
        return X_test, y_test, d


# 仅仅是二分类使用
def read_teeth_data(path, **kwargs):
    # path 是牙齿文件根目录 不是txt的路径
    # 例如 path=r'D:\desktop\test_code'
    # return: 输出迭代器 每一次迭代返回牙齿的号位 分割前的牙齿数据 分割后的牙齿数据
    predict = False
    is_raw = True
    four_class_seg = False
    if 'predict' in kwargs:
        predict = kwargs['predict']
    if 'is_raw' in kwargs:
        is_raw = kwargs['is_raw']

    def __read(lst_d, for_all=False, decimals=3):
        # decimals表示保留小数点后几位 由于牙齿数据的前30个有精度问题 所以现在统一把精度改为3
        # 确保processing_data.py的__label_data方法可以打上正确的标签
        # 实验证明decimals改为3是有效的
        all = []
        for l in lst_d:
            n_path = os.path.join(path1, l)  # txt
            all_ = np.loadtxt(n_path)
            if not for_all:
                if is_raw:
                    l_int = eval(l.strip('.txt')[1])  # '34' -> 4
                    all = [l_int, all_]
                else:
                    l_int = 1
                    all = [l_int, all_]
            else:
                all = all_
        if not for_all:
            all[1] = np.round(all[1], decimals=decimals)  # all[1]是点数据
        else:  # for all
            all = np.round(all, decimals=decimals)
        return all

    p = os.listdir(path)
    res = []  # 每个分割出来的牙齿 eg.[36,[...(points)]]
    all = []  # 没有分割的半口牙

    if is_raw:
        for i in p:  # person
            person = []
            path1 = os.path.join(path, i)
            lst_d = os.listdir(path1)
            print(path1)
            if lst_d is None:
                raise ValueError('No content in this file')
            '''这里读取4 5 6 7号牙 不区分上下牙颌'''

            l_concat4 = list(filter(lambda x: x.endswith('4.txt'), lst_d))
            l_concat5 = list(filter(lambda x: x.endswith('5.txt'), lst_d))
            l_concat6 = list(filter(lambda x: x.endswith('6.txt'), lst_d))
            l_concat7 = list(filter(lambda x: x.endswith('7.txt'), lst_d))
            l_all = list(filter(lambda x: x.startswith('all'), lst_d))

            l4 = __read(l_concat4)
            l5 = __read(l_concat5)
            l6 = __read(l_concat6)
            l7 = __read(l_concat7)
            l_a = __read(l_all, True)

            try:
                if len(l_a) == 0:
                    raise RuntimeError
                else:
                    all.append(l_a)
            except RuntimeError:
                print("Don't Exist 'all.txt' in This File")

            if len(l4) != 0:
                person.append(l4)
            if len(l5) != 0:
                person.append(l5)
            if len(l6) != 0:
                person.append(l6)
            if len(l7) != 0:
                person.append(l7)
            res.append(person)

    else:  # is_raw==False
        for i in p:
            person = []
            path1 = os.path.join(path, i)
            lst_d = os.listdir(path1)
            print(path1)
            if lst_d is None:
                raise ValueError('No content in this file')
            l_concat1 = ['1.txt']
            l_all = ['rest.txt']

            l1 = __read(l_concat1)
            l_a = __read(l_all, for_all=True)

            try:
                if len(l_a) == 0:
                    raise RuntimeError
                else:
                    all.append(l_a)
            except RuntimeError:
                print("Exist No File")

            if len(l1) != 0:
                person.append(l1)
            if len(l_a) != 0:
                person.append(l_a)
            res.append(person)

    if not predict:
        rate = kwargs['test_rate']
        X_train, X_test, y_train, y_test = train_test_split(res, all, test_size=rate)
        return iter(X_train), iter(X_test), y_train, y_test
    else:  # just predict
        return iter(res), all


# TODO: implement
# TODO: Need Test
def read_teeth_data_five_class(path, *args, **kwargs):
    # eg. root dir
    #    |- 001 -|- 34.txt
    #            |- 35.txt
    #            |- 36.txt
    #            |- 37.txt
    #            |- bg.txt   (use in five class seg)
    #   |- 002 -|- ...
    """
    四分类读入函数
    :param path:根目录
    :param kwargs: 其他参数
    :return:
    """
    all_teeth = []
    decimals = None
    if 'decimals' in kwargs:
        decimals = kwargs['decimals']
    file_lst = os.listdir(path)
    _all = []
    for p in file_lst:  # 每半口牙
        ins_teeth = []
        ins_path = os.path.join(path, p)
        print(ins_path)
        four_teeth_lst = os.listdir(ins_path)  # .txt文件
        assert len(four_teeth_lst) == 5

        for t in four_teeth_lst:
            if t != 'bg.txt':  # 去除bg.txt
                teeth_path = os.path.join(ins_path, t)
                label = eval(t.strip('.txt'))
                teeth = np.loadtxt(teeth_path)
                assert teeth.shape[1] == 3
                # one2four_teeth.append(teeth)  # 背景点去除 # list type
                if decimals is not None:
                    teeth = np.round(teeth, decimals=decimals)
                ins_teeth.append([label, teeth])
            else:  # 'bg.txt'
                bg = np.loadtxt(os.path.join(ins_path, 'bg.txt'))
                if decimals is not None:
                    bg = np.round(bg, decimals=decimals)
                _all.append(bg)
        all_teeth.append(ins_teeth)
    return all_teeth, _all


def read_teeth_data_four_class(path, **kwargs):
    # eg. root dir
    #    |- 001 -|- 34.txt
    #            |- 35.txt
    #            |- 36.txt
    #            |- 37.txt
    #            |- all.txt   (no use in four class seg)
    #   |- 002 -|- ...
    """
    四分类读入函数
    :param path:根目录
    :param kwargs: 其他参数
    :return:
    """
    all_teeth = []
    decimals = None
    if 'decimals' in kwargs:
        decimals = kwargs['decimals']
    file_lst = os.listdir(path)
    _all = []
    for p in file_lst:  # 每半口牙
        ins_teeth = []
        ins_path = os.path.join(path, p)
        print(ins_path)
        four_teeth_lst = os.listdir(ins_path)  # .txt文件
        # TODO: 6/14 upload to the remote
        assert len(four_teeth_lst) == 5 or 4  # 或者没有.txt文件

        # one2four_teeth = []

        for t in four_teeth_lst:
            if t != 'all.txt':  # 去除all.txt
                teeth_path = os.path.join(ins_path, t)
                label = eval(t.strip('.txt')[1]) - 4  # 37.txt -> 37 -> 7 -> 3
                teeth = np.loadtxt(teeth_path)
                assert teeth.shape[1] == 3
                # one2four_teeth.append(teeth)  # 背景点去除 # list type
                if decimals is not None:
                    teeth = np.round(teeth, decimals=decimals)
                ins_teeth.append([label, teeth])
            else:  # 'all.txt'
                alli = np.loadtxt(os.path.join(ins_path, 'all.txt'))
                if decimals is not None:
                    alli = np.round(alli, decimals=decimals)
                _all.append(alli)
        all_teeth.append(ins_teeth)
    return all_teeth, _all


if __name__ == '__main__':
    # iter, all = read_teeth_data(r'D:\desktop\3', predict=True, is_raw=True)

    all_teeth = read_teeth_data_four_class('D:/desktop/3')
    pass
