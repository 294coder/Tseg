import numpy as np
import open3d as o3d
from get_division_polylines import array_to_pointcloud
from load_data import load_data_txt
from Three_registration import process_icp_usual


def in_convex_polyhedron(points_set: np.ndarray, test_points: np.ndarray):
    """
    检测点是否在凸包内
    :param points_set: 凸包，需要对分区的点进行凸包生成 具体见conv_hull函数
    :param test_points: 检测点
    :return: bool类型
    """
    assert type(points_set) == np.ndarray
    assert type(points_set) == np.ndarray
    bol = np.zeros((test_points.shape[0], 1), dtype=np.bool)
    ori_set = points_set
    _, ori_edge_index = conv_hull(ori_set)
    ori_edge_index = np.sort(np.unique(ori_edge_index))
    for i in range(test_points.shape[0]):
        new_set = np.concatenate((points_set, test_points[i, np.newaxis]), axis=0)
        _, new_edge_index = conv_hull(new_set)
        new_edge_index = np.sort(np.unique(new_edge_index))
        bol[i] = (new_edge_index.tolist() == ori_edge_index.tolist())
    return bol


def conv_hull(points: np.ndarray):
    """
    生成凸包 参考文档：https://blog.csdn.net/io569417668/article/details/106274172
    :param points: 待生成凸包的点集
    :return: 索引 list
    """
    pcl = array_to_pointcloud(points)
    hull, lst = pcl.compute_convex_hull()
    return hull, lst


if __name__ == '__main__':
    # test
    # path = r'D:\desktop\data\lower_jaw_data\points1.txt'
    # points_set = np.array(load_data_txt(path))
    # p = np.array([[2.31740, -0.72062, 12.51270], [115, 115, 115], [2.31740, -0.72062, 19.51270]])
    # print(in_convex_polyhedron(points_set, p))  # True, False, False

    source_path = r'D:\desktop\teeth_6_note\ply\1seg.ply'
    target_path = r'D:\desktop\teeth_6_note\ply\2seg.ply'


    pcd1 = o3d.io.read_point_cloud(source_path)
    pcd2 = o3d.io.read_point_cloud(target_path)
    pcd1.paint_uniform_color([0, 0, 1])
    pcd2.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries([pcd1, pcd2])

    m=process_icp_usual(source_path,target_path,more_accuracy=True)
    pcd1=pcd1.transform(m)
    o3d.visualization.draw_geometries([pcd1, pcd2])

