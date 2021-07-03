import numpy as np
import open3d as o3d
import copy
import Three_registration as tre
import get_division_polylines as gdp
import get_stl_points_and_mesh as gsm
import load_data as ld
import pointnet.models.pointnet
import pointnet.models.pointnet_part_seg
import pointnet.models.pointnet_sem_seg
from get_division_polylines import points_index  # 类
import sys
sys.path.append('..')
sys.path.append(r'D:/codeblock/code/dachuang/pointnet')


def draw_3d_teeth_pic(teeth, *args):
    o3d.visualization.draw_geometries([teeth, *args])


def trans(filename):
    source_mesh = o3d.io.read_triangle_mesh(filename)
    # 转换为ply后可视化
    o3d.io.write_triangle_mesh(r"D:\desktop\Polyline.ply", source_mesh)  # 将stl格式转换为ply格式
    source_ply = o3d.io.read_point_cloud(r"D:\desktop\Polyline.ply")
    return source_ply


if __name__ == '__main__':
    # gsm.get_stl_points_meshes_data(4)
    mesh_path = r'D:/desktop/data_1/lower_jaw_data/meshes9.txt'
    points_path = r'D:/desktop/data_1/lower_jaw_data/points9.txt'
    gdp.get_division_polylines_class_process(mesh_path, points_path, 8000, 15000)
    # gdp.get_division_polylines(mesh_path, points_path)

'''
point_cloud_file_path = r'D:\\desktop\\result_icp_ptp.ply'
# 读取点云
pcd = o3d.io.read_point_cloud(point_cloud_file_path)
tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
pcd.estimate_normals()
o3d.visualization.draw_geometries([pcd])'''

'''
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
    pcd, .7, tetra_mesh, pt_map)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)'''
'''
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd,9)
o3d.visualization.draw_geometries([mesh])'''
'''
radii = [0.005, 0.01, 0.02, 0.04]
rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
               pcd, o3d.utility.DoubleVector(radii))
o3d.visualization.draw_geometries([rec_mesh])'''
