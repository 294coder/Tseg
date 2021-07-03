import open3d as o3d
from typing import Union, List
import numpy as np
import torch


def array_to_pointcloud(*np_array) -> Union[o3d.geometry.PointCloud, List[o3d.geometry.PointCloud]]:
    if len(np_array) > 1:
        all_pcd = []
        for array in np_array:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(array)
            pcd.compute_point_cloud_distance()
            all_pcd.append(pcd)
    else:  # ==1
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np_array[0])
        all_pcd = pcd
    return all_pcd


def compute_pcd_distance(arr1, arr2):
    # arr1 is source, arr2 is target
    source, target = array_to_pointcloud([arr1, arr2])

    hull, _ = target.compute_convex_hull()
    np_hull = np.asarray(hull.vertices)
    hull = array_to_pointcloud(np_hull)

    d = source.compute_point_cloud_distance(hull)
    return d


# noinspection PyShadowingBuiltins
# for two seg teeth
def module_predict_visualization(all: torch.Tensor, labels: torch.Tensor, for4class=False):
    color_all = [[.5, .5, .5], [0, 0, 1], [0, 1, 0], [1, 0, 0], [.2, .4, .8], [1, 1, .3],
                 [.2, .7, 1], [.5, 1, 1], [.4, .5, .6]]
    assert len(all) == len(labels)
    # type invert   tensor->ndarray
    all = all.cpu().numpy()
    labels = labels.cpu().numpy()

    bg = array_to_pointcloud(all[np.argwhere(labels == 0)].squeeze())  # background
    t1 = array_to_pointcloud(all[np.argwhere(labels == 1)].squeeze())
    if for4class:
        t2 = array_to_pointcloud(all[np.argwhere(labels == 2)].squeeze())
        t3 = array_to_pointcloud(all[np.argwhere(labels == 3)].squeeze())
        t2.paint_uniform_color(color_all[2])
        t3.paint_uniform_color(color_all[3])

    # uniform color
    bg.paint_uniform_color(color_all[0])
    t1.paint_uniform_color(color_all[1])
    if for4class:
        return bg, t1, t2, t3
    else:
        return bg, t1
    # o3d.visualization.draw_geometries([bg, t1])


def make_boundingbox(pcd: Union[o3d.geometry.PointCloud, np.ndarray]):
    input_type = type(pcd)
    if input_type == o3d.geometry.PointCloud:
        bounding_box = pcd.get_axis_aligned_bounding_box()
        bounding_box.color = (0, 1, 0)
        return bounding_box
    if input_type == np.ndarray:
        pcd = array_to_pointcloud(pcd)
        bounding_box = pcd.get_axis_aligned_bounding_box()
        bounding_box.color = (0, 0, 1)
        return bounding_box


def bounding_box_visualization(pcd, pred, for_4class=True):
    t_4 = list(module_predict_visualization(pcd, pred, for4class=for_4class))
    teeth_and_bdb = t_4.copy()
    for t in t_4:
        teeth_and_bdb.append(make_boundingbox(t))
    o3d.visualization.draw_geometries(teeth_and_bdb)
