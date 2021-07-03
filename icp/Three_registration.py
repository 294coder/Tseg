import open3d as o3d
import time
import numpy as np
import copy
from utils.open3d_utils import array_to_pointcloud


# ICP配准
def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
          % distance_threshold)
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


# 画图
def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)  # transformation是变换矩阵（旋转矩阵和平移矩阵）
    o3d.visualization.draw_geometries([source_temp, target])


# 画图
# def draw_registration_result(source, target, transformation):
#     source_temp = copy.deepcopy(source)
#     target_temp = copy.deepcopy(target)
#     # 上色
#     source_temp.paint_uniform_color([1, 0.706, 0])
#     target_temp.paint_uniform_color([0, 0.651, 0.929])
#     source_temp.transform(transformation)
#     o3d.visualization.draw_geometries([source_temp, target_temp])


# 预处理，降采样，计算FPFH特征
def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def point_cloud_format_trans(path):
    '''这是我自己写的函数
    source_path和target是输入文件和输出文件的路径
    需要改一下 改成你自己电脑上的文件存储位置
    source_mesh是读入stl格式后的中间文件
    他会在你给的stl文件夹内生成一个ply格式的文件
    然后o3d会读入这个中间文件
    最后再返回source跟target
    注意 是点云形式'''
    source_mesh = o3d.io.read_triangle_mesh(path)
    # 转换为ply后可视化
    o3d.io.write_triangle_mesh(r"D:\desktop\data_ply.ply", source_mesh)  # 将stl格式转换为ply格式
    source_ply = o3d.io.read_point_cloud(r"D:\desktop\data_ply.ply")

    return source_ply


def template_estimate_normals(template):
    template.estimate_normals(template,
                              search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # o3d.visualization.draw_geometries(template)


def prepare_dataset(source_path, target_path, voxel_size=.2):
    # type check
    if isinstance(source_path, str) and isinstance(target_path, str):
        source_path_ends = source_path.split('.')[1]
        target_path_ends = target_path.split(".")[1]
        path = [source_path, target_path]
        st = []
        print(":: 初始情况")
        for i, ends in enumerate([source_path_ends, target_path_ends], 0):
            if ends == 'stl':
                st.append(point_cloud_format_trans(path[i]))
            elif ends == 'ply':
                st.append(o3d.io.read_point_cloud(path[i]))
            elif ends == 'txt':
                pcd_txt = np.loadtxt(path[i])
                st.append(array_to_pointcloud(pcd_txt))
            else:
                raise TypeError(" path type error, should end with 'stl','ply' or 'txt' ")
        source, target = st
    elif isinstance(source_path, o3d.geometry.PointCloud) and isinstance(target_path, o3d.geometry.PointCloud):
        source, target = source_path, target_path
    else:
        raise TypeError(" input type error augment #1 and #2 should be str or o3d.gemoetry.PointCloud")

    # 初始变换矩阵
    trans_init = np.asarray([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
    source.transform(trans_init)
    #    draw_registration_result(source, target, np.identity(4))
    o3d.geometry.PointCloud.estimate_normals(source,
                                             search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    o3d.geometry.PointCloud.estimate_normals(target,
                                             search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    return source, target, source_down, target_down, source_fpfh, target_fpfh


def icp_global(source, target, source_down, target_down, source_fpfh, target_fpfh, voxel_size=0.2):
    # 开始准备
    current_transformation = np.asarray([[1, 0, 0, 0],
                                         [0, 1, 0, 0],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 1]])
    draw_registration_result_original_color(source, target, current_transformation)
    # 全局
    start = time.time()
    result_fast = execute_fast_global_registration(source_down, target_down,
                                                   source_fpfh, target_fpfh,
                                                   voxel_size)
    print("Fast global registration took %.3f sec.\n" % (time.time() - start))
    print(result_fast)
    print(result_fast.transformation)
    draw_registration_result_original_color(source_down, target_down, result_fast.transformation)
    return result_fast


def icp_ptp(source, target, result_fast, threshold=1):
    # 点对点
    print(":: point to point icp")
    result_icp_ptp = o3d.pipelines.registration.registration_icp(
        source, target, threshold, result_fast.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(result_icp_ptp)
    print(result_icp_ptp.transformation)
    draw_registration_result_original_color(source, target, result_icp_ptp.transformation)
    return result_icp_ptp


# 这个函数原先没用
def icp_ptplane(source, target, current_transformation, threshold=1):
    # point-to-plane icp
    print("point-to-plane icp")
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, threshold, current_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    print(result_icp)
    print(result_icp.transformation)
    draw_registration_result_original_color(source, target, result_icp.transformation)
    return result_icp


# color
'''
    print("colored point cloud配准")
    result_icp2 = o3d.pipelines.registration.registration_icp(
        source, target, threshold, current_transformation,
        o3d.pipelines.registration.TransformationEstimationForColoredICP())
    print(result_icp2)
    print(result_icp2.transformation)
    draw_registration_result_original_color(source, target, result_icp2.transformation)'''


def save_result(source, target, template_path, trans_matrix):
    source_copy = copy.deepcopy(source)
    source_copy.transform(trans_matrix)
    pcd = o3d.geometry.PointCloud()

    s_points = np.asarray(source_copy.points)
    t_points = np.asarray(target.points)

    s_normals = np.asarray(source_copy.normals)
    t_normals = np.asarray(target.normals)

    st_points = np.concatenate([s_points, t_points], axis=0)
    st_normals = np.concatenate([s_normals, t_normals], axis=0)

    pcd.points = o3d.utility.Vector3dVector(st_points)
    # pcd.normals = o3d.utility.Vector3dVector(st_normals)

    # estimate_normals计算发现 但是没有使用write_triangle_mesh所以这句话没用
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # 这句话会把面片信息丢掉
    o3d.io.write_point_cloud(template_path, pcd)


def process_icp(source_path, target_path, template_path, i):
    source, target, source_down, target_down, source_fpfh, target_fpfh = \
        prepare_dataset(source_path, target_path)
    result_global = icp_global(source, target, source_down, target_down, source_fpfh, target_fpfh)
    result_icp_ptp = icp_ptp(source, target, result_global)
    save_result(source_down, target_down, template_path=template_path, trans_matrix=result_icp_ptp.transformation)
    print('::------第%d次模板覆盖完成------' % (i - 1))


def process_icp_usual(source_path, target_path, save=False,
                      template_path=r'D:\\desktop\\template.ply', ptplane=False):
    result_icp_ptplane = None
    source, target, source_down, target_down, source_fpfh, target_fpfh = \
        prepare_dataset(source_path, target_path)
    result_global = icp_global(source, target, source_down, target_down, source_fpfh, target_fpfh)
    result_icp_ptp = icp_ptp(source, target, result_global)
    if ptplane:
        result_icp_ptplane = icp_ptplane(source, target, result_icp_ptp.transformation)

    if save:
        if result_icp_ptplane is not None:
            matrix = result_icp_ptplane.transformation
        else:
            matrix = result_icp_ptp.transformation
        save_result(source_down, target_down, template_path=template_path,
                    trans_matrix=matrix)

    return result_icp_ptp.transformation


if __name__ == '__main__':
    source_path = r'D:\desktop\teeth_program_read\rotate（51-55）\054-lu-r\all.txt'  # 缺损
    target_path = r"D:\desktop\teeth_program_read\rotate（1-40）\005-lu-r\all.txt"
    process_icp_usual(source_path, target_path)
