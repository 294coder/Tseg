import open3d as o3d
import numpy as np
import Three_registration as tre


def draw_3d_pic(template):
    o3d.visualization.draw_geometries([template])


def trans(filename):
    source_mesh = o3d.io.read_triangle_mesh(filename)
    # 转换为ply后可视化
    o3d.io.write_triangle_mesh(r"D:\desktop\Polyline.ply", source_mesh)  # 将stl格式转换为ply格式
    source_ply = o3d.io.read_point_cloud(r"D:\desktop\Polyline.ply")
    return source_ply


if __name__ == '__main__':

    path_template = r"D:\desktop\result_icp.ply"
    path_low_target = r'D:\desktop\teeth6\L1.ply'
    for i in np.arange(2, 4):
        path_low_source = r'D:\desktop\teeth6' + r'\L' + str(i) + '.ply'
        # path_upper_source = r'D:\desktop\teeth6' + r'\U' + str(i) + '.stl'
        tre.process_icp(source_path=path_low_source, target_path=path_low_target, template_path=path_template,
                        i=i, format_trans=False)

        # ------------------------画出模板图像------------------------
