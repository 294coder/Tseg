from utils.open3d_utils import array_to_pointcloud
import numpy as np
import open3d as o3d


def check_cover_point(pcd1, pcd2, d=0.1) -> (o3d.geometry.PointCloud, o3d.geometry.PointCloud):
    pcd1 = array_to_pointcloud(pcd1)
    pcd2 = array_to_pointcloud(pcd2)
    distance1 = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
    distance2 = np.asarray(pcd2.compute_point_cloud_distance(pcd1))

    idx1 = np.where(distance1 < d)[0]
    idx2 = np.where(distance2 < d)[0]
    outer_idx1 = np.where(distance1 > d)[0]
    outer_idx2 = np.where(distance2 > d)[0]
    cover_pts1 = pcd1.select_by_index(idx1)
    cover_pts2 = pcd2.select_by_index(idx2)
    rest_pts1 = pcd1.select_by_index(outer_idx1)
    rest_pts2 = pcd2.select_by_index(outer_idx2)

    return cover_pts1, cover_pts2, rest_pts1, rest_pts2


if __name__ == "__main__":
    path1 = 'D:/desktop/001-ll-36.txt'
    path2 = 'D:/desktop/001-lu-26.txt'
    pcd1 = np.loadtxt(path1)
    pcd2 = np.loadtxt(path2)
    pcd1_select, pcd2_select, out_pcd1, out_pcd2 = check_cover_point(pcd1, pcd2)

    pcd1_select.paint_uniform_color([0, 0, 1])
    pcd2_select.paint_uniform_color([0, 0, 1])
    out_pcd1.paint_uniform_color([0, 1, 0])
    out_pcd2.paint_uniform_color([0, 1, 0])

    o3d.visualization.draw_geometries([out_pcd2, pcd2_select, out_pcd1, pcd1_select])
