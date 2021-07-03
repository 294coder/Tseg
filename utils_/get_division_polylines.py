import numpy as np
import torch
import math
import open3d as o3d
import copy
import matplotlib.pyplot as plt
import time


def angle_of_vector(v1, v2):
    A = np.linalg.norm(v1 * v2)
    B = np.linalg.norm(v1) * np.linalg.norm(v2)
    return math.acos(A / B) * 180 / np.pi


# 将分割线着重标出
def emphasize_division_polylines(points_class, teeth_points, poly_points_index, dbscan_left, dbscan_right,
                                 draw_line=False, only_poly_points=True):
    poly_points = teeth_points[poly_points_index].squeeze()
    print('::get poly_points')
    polylines = pcd_dbscan(poly_points, dbscan_left, dbscan_right)
    print('::get DBSCAN done')

    teeth_points = np.delete(teeth_points, poly_points_index, axis=0)
    teeth = array_to_pointcloud(teeth_points)

    def _prepare_for_linesets(now_point, poly_points_index):

        assert now_point.new_index != None

        idx = now_point.new_index
        lines = []
        for con_point in now_point.connect_points:
            if con_point in poly_points_index:
                lines.append([idx, con_point])
        if len(lines) != 0:
            return lines

    def _create_new_points_class(raw_points_class, poly_points_index):
        num = poly_points_index.size
        new_points_class_list = []
        for i in range(num):
            rpc = raw_points_class[poly_points_index[i]]
            idx = rpc.index
            point = points_index(idx, i)  # 创建新的点
            l = []
            for cnp in rpc.connect_points:  # 重新映射index
                nidx = np.where(poly_points_index == cnp)
                if len(nidx[0]) != 0:
                    l.append(nidx[0].item())
            new_connect_points_index = np.array(l)
            point.connect_points = new_connect_points_index
            new_points_class_list.append(point)
        return new_points_class_list

    # 这个本来是去除线的函数 但是重复的线是不影响效果的 反而调用这个函数会增加运行时间 不要调用它
    def _get_repeat_lines_off(lines):
        unrepeat_lines = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                if (lines[i] == lines[j]) or ([lines[i][1], lines[i][0]] == lines[j]):
                    break
                if j == len(lines) - 1:
                    unrepeat_lines.append(lines[i])
        unrepeat_lines.append(lines[len(lines) - 1])
        return unrepeat_lines

    def draw_line_func():
        lines = []
        new_points_class = _create_new_points_class(points_class, poly_points_index)
        print('::get new_points_list constructed')

        for point in new_points_class:
            line = _prepare_for_linesets(point, poly_points_index)
            if line:
                for elem in line:
                    lines.append(elem)

        lines = np.array(lines).squeeze()

        line_set = get_line_set(poly_points, lines)
        print('::get line_set')
        return line_set

    teeth.paint_uniform_color([1, 0.706, 0])
    # polylines.paint_uniform_color(color)

    if draw_line:
        line_set = draw_line_func()
        o3d.visualization.draw_geometries([polylines, line_set])
        print('::draw geometry')
    elif only_poly_points:
        o3d.visualization.draw_geometries([polylines])
        print('::draw geometry')
    else:
        o3d.visualization.draw_geometries([teeth, polylines])
        print('::draw geometry')


def pcd_dbscan(dot_special: np.ndarray, eps: float = .85, min_points: int = 25):  # 返回Pointcloud类型
    pcd = array_to_pointcloud(dot_special)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    max_labels = labels.max()
    print('::DBSCAN has found %d cluster(s)' % (max_labels + 1))
    colors = plt.get_cmap("tab20")(labels / (max_labels if max_labels > 0 else 1))
    colors[labels < 0] = 0  # 噪声点为黑色
    # 删除噪声点
    # remain_idx=[labels!=-1]
    # dot_special=dot_special[tuple(remain_idx)]
    # pcd = array_to_pointcloud(dot_special)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return pcd


def get_line_set(points, lines):
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def array_to_pointcloud(np_array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_array)
    return pcd


def estimate_normals(pcd):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(.1, 30))
    return np.asarray(pcd.normals)


# 废物类 爬
class points_index():
    def __init__(self, index, *args):
        if len(args) == 0:
            self.connect_points = np.array([], dtype=np.int)
        elif len(args) == 2:
            self.connect_points = args[0]
        else:
            self.new_index = args[0]  # 画线中的引索
        self.index = index  # 实际点云中的引索
        self.normals = None

        # PointNet
        self.coordinate = None
        self.cls = False

    # 不要用这个巨慢的函数
    def _search_connect(self, mesh_dot):
        for triangle in mesh_dot:
            for j in range(3):
                if triangle[j] == self.index:
                    const = np.array([0, 1, 2])
                    const = np.delete(const, j)
                    elems = triangle[const]
                    self.append_unique(elems)

    def append_unique(self, elems):
        for elem in elems:
            if elem not in self.connect_points:
                self.connect_points = np.append(self.connect_points, elem)


# 跑的太尼玛慢了 爬
# 这个真的跑的巨他妈的慢
# 不愧是我
# 这么烂的代码
# 懒得维护了
def get_division_polylines_class_process(mesh_path, points_path, left_interval,
                                         right_interval, dbscan_left_interval, dbscan_right_interval, ignore=True):
    if ignore:
        np.seterr(divide='ignore', invalid='ignore')

    start = time.time()
    mesh_dot = np.loadtxt(mesh_path, dtype=np.int)
    dot_xyz_f = np.loadtxt(points_path, dtype=np.float)
    dot_xyz = np.array(dot_xyz_f)
    K_list = []  # 这个是每个点的平均曲率的值
    points = []

    idx = np.unique(mesh_dot)
    assert idx.shape[0] == dot_xyz_f.shape[0]
    for i in idx:
        points.append(points_index(i))
    print('::get points_list constructed')

    for triangle in mesh_dot:
        points[triangle[0]].append_unique([triangle[1], triangle[2]])
        points[triangle[1]].append_unique([triangle[0], triangle[2]])
        points[triangle[2]].append_unique([triangle[0], triangle[1]])
    print("::get every point's connected points")

    pcd = array_to_pointcloud(dot_xyz_f)
    normals = estimate_normals(pcd)
    print('::get vertices normals')

    for point in points:

        point.normals = normals[point.index]  # ? 这里有疑问

        i = point.index
        my_dictl = point.connect_points

        N_average = point.normals
        K_group = []

        for m in range(point.connect_points.size):
            K_group.append(2 * angle_of_vector(N_average, dot_xyz[my_dictl[m]] - dot_xyz[i]) / (
                    np.linalg.norm(dot_xyz[my_dictl[m]] - dot_xyz[i]) ** 2))
        K_list.append((max(K_group) + min(K_group)) / 2)
    K_list_array = np.array(K_list)

    dot_special = np.where((K_list_array > left_interval) & (K_list_array < right_interval))  # 查找曲率值大于2000的值的位置
    dot_special = np.array(dot_special).squeeze()
    print('::get points between the interval')
    print('::start to emphasize the point cloud picture')
    emphasize_division_polylines(points, dot_xyz, dot_special, dbscan_left_interval, dbscan_right_interval,
                                 draw_line=False, only_poly_points=True)
    print('::whole function consume %.2f s' % (time.time() - start))
    print('我说吧 这个函数巨他妈的慢')
    return dot_special


# 去掉z轴 画二维图像
def get_teeth_2d(mesh_path, points_path):
    mesh = np.loadtxt(mesh_path, dtype=int)
    points = np.loadtxt(points_path, dtype=np.float)
    points_xy = points[:, 0:2]
    plt.scatter(points_xy[:, 0].reshape(1, -1), points_xy[:, 1].reshape(1, -1), s=1)
    plt.show()


# 这个是孙新武写的
def get_division_polylines(mesh_path, points_path, ignore=True):
    if ignore:
        np.seterr(divide='ignore', invalid='ignore')
    mesh_dot = np.loadtxt(mesh_path, dtype=np.int)
    print(len(mesh_dot))
    my_dict = dict()  # 建立一个字典集，每个key对于一个点，key的值表示这个点和哪些点直接相连
    K_list = []  # 这个是每个点的平均曲率的值
    for i in range(len(mesh_dot)):  # 这个for函数是将字典集建立起来，通过遍历的方式，找出每个点所连接的点
        for j in range(3):
            if j == 0:
                my_dict.setdefault(mesh_dot[i][j], set()).add(mesh_dot[i][j + 1])
                my_dict.setdefault(mesh_dot[i][j], set()).add(mesh_dot[i][j + 2])
            elif j == 1:
                my_dict.setdefault(mesh_dot[i][j], set()).add(mesh_dot[i][j - 1])
                my_dict.setdefault(mesh_dot[i][j], set()).add(mesh_dot[i][j + 1])
            else:
                my_dict.setdefault(mesh_dot[i][j], set()).add(mesh_dot[i][j - 2])
                my_dict.setdefault(mesh_dot[i][j], set()).add(mesh_dot[i][j - 1])

    dot_xyz_f = np.loadtxt(points_path, dtype=np.float)  # 获取点的坐标，这里是list的格式，下面要转换为向量的形式
    dot_xyz = np.array(dot_xyz_f)
    print(len(my_dict))

    for i in range(len(my_dict)):  # 字典集的长度也就是点的个数
        N_fk = list()  # 这是一个中间参量，求与顶点相连的每个三角形的法向量
        my_dictl = list(my_dict[i])  # 之前为了字典里面一个key对多个值，将值设置为集合形式，这里改为list形式
        for j in range(len(my_dictl)):  # 这一部分为求三角形法向量的过程
            if j != (len(my_dictl) - 1):
                a_1 = (dot_xyz[i] - dot_xyz[my_dictl[j + 1]]) * (dot_xyz[my_dictl[j + 1]] - dot_xyz[my_dictl[j]])
            else:
                a_1 = (dot_xyz[i] - dot_xyz[my_dictl[0]]) * (dot_xyz[my_dictl[0]] - dot_xyz[my_dictl[j]])

            N_fk.append(a_1 / np.linalg.norm(a_1))
        N_average = sum(np.array(N_fk)) / len(my_dictl)  # N_fk之前为list形式，这里转换为向量形式，这里是求顶点的平均法向量。
        K_group = []  # 获得点的曲率值，便于找最大值和最小值

        for m in range(len(my_dictl)):
            K_group.append(2 * sum(N_average * (dot_xyz[my_dictl[m]] - dot_xyz[i])) / (
                    np.linalg.norm(dot_xyz[my_dictl[m]] - dot_xyz[i]) ** 2))
        K_list.append((max(K_group) + min(K_group)) / 2)
    K_list_array = np.array(K_list)  # 换个格式
    dot_special = np.argwhere(abs(K_list_array) > 6)  # 查找曲率值大于2000的值的位置
    print(len(dot_special))
    emphasize_division_polylines(dot_xyz, dot_special)
    return dot_special


if __name__ == '__main__':
    # 可以改的参数是8000 15000 0.85 17
    # 下面的path是网格和点
    mesh_path = r'D:/desktop/data_1/lower_jaw_data/meshes9.txt'
    points_path = r'D:/desktop/data_1/lower_jaw_data/points9.txt'
    get_division_polylines_class_process(mesh_path, points_path, 8000, 15000, 0.85, 17)
