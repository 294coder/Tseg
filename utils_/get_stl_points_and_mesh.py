import os
import time

import numpy as np
import open3d as o3d

gb = r'D:\\desktop'  # 'C:\\desktop'
'''
txt文件默认生成在桌面
建议使用 cloud compare 生成的ply文件传入r
open3d生成的ply跟软件生成的有差异
如果都行的话我把这个程序写成全自动的
已经测试过了 ply文件直接打开跟本程序生成的点坐标和面引索完全一致
我的电脑的桌面在D盘 如果不一致请修改gb变量
'''


# 得到点的坐标和组成三角形的面的标号
def get_mesh_data(mesh, remove_dupliacte_points=False):
    if remove_dupliacte_points:
        o3d.geometry.TriangleMesh.remove_duplicated_vertices(mesh)
        o3d.geometry.TriangleMesh.remove_duplicated_triangles(mesh)
    meshes = np.asarray(mesh.triangles)
    points = np.asarray(mesh.vertices)
    return points, meshes


# 使用open3d来完成stl格式的转换 与 cloudcompare 转换有出入
def _trans(source_path):
    source_mesh = o3d.io.read_triangle_mesh(source_path)
    '''我在桌面生成一个缓存文件 这个文件是ply格式的'''
    o3d.io.write_triangle_mesh(r"D:\desktop\source_mesh_raw_data.ply", source_mesh)  # 将stl格式转换为ply格式
    source_ply_mesh = o3d.io.read_triangle_mesh(r"D:\desktop\source_mesh_raw_data.ply")
    return source_ply_mesh


def write_to_txt(path, points_data, meshes_data, num):
    # num对应每个牙的标号
    with open(path + r'points' + str(num) + '.txt', 'a+')as pd:
        with open(path + r'meshes' + str(num) + '.txt', 'a+')as md:
            for elem1 in points_data.squeeze():
                pd.write(str(elem1).strip('[').strip(']') + '\n')
            for elem2 in meshes_data.squeeze():
                md.write(str(elem2).strip('[').strip(']') + '\n')
    pd.close()
    md.close()
    print('::写入成功')


# 如果传进来的是原文件 就调用这个函数
def get_stl(path, low_jaw, num):
    print('::注意 源文件传入 和软件转换后的坐标不一致')
    spm = _trans(source_path=path)  # 转换格式
    points, meshes_index = get_mesh_data(spm, True)
    if low_jaw:
        output_path = gb + r'\\data\\lower_jaw_data\\'
    else:
        output_path = gb + r'\\data\\upper_jaw_data\\'
    write_to_txt(output_path, points, meshes_index, num)


# 如果传进来的是cloud compare生成的ply文件
def get_cloud_compare(path):
    print('::软件导出文件传入')
    spm = o3d.io.read_triangle_mesh(path)
    points, meshes_index = get_mesh_data(spm, True)
    write_to_txt(points, meshes_index)


# 新建文件夹 不需要手动调用
def _make_dir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print('::文件夹不存在 开始新建文件夹')
        print('::新建文件夹成功')

    else:
        print('::文件夹存在 不需要创建文件夹')


def get_stl_points_meshes_data(file_num):
    '''
    如果你使用的是软件传入的文件请使用 get_cloud_compare 函数 
    如果你使用的是源文件就使用 get_stl 函数
    '''
    _make_dir(gb + r'\\data')  # 在桌面上创建文件夹 准备存入数据
    _make_dir(gb + r'\\data\\lower_jaw_data')
    _make_dir(gb + r'\\data\\upper_jaw_data')

    start = time.time()
    # 开始得到下牙的数据
    for i in np.arange(1, file_num):
        # 这里的牙齿文件需要重命名
        path_low_source = gb + r'\\teeth6' + r'\\L' + str(i) + '.ply'
        start1 = time.time()
        low_jaw = True
        get_stl(path_low_source, low_jaw=low_jaw, num=i)
        print('::第 %d 个下牙数据输出成功' % i)
        print('::运行时长 %.2f s' % (time.time() - start1))
    '''
    # 开始得到上牙数据
    for i in np.arange(1, file_num):
        path_upper_source = gb + r'\\teeth_data' + r'\\U' + str(i) + '.stl'
        start2 = time.time()
        low_jaw = False
        get_stl(path_upper_source, low_jaw=low_jaw, num=i)
        print('::第 %d 个上牙数据输出成功' % i)
        print('::运行时长 %.2f s' % (time.time() - start2))
    '''
    print('::程序运行结束 运行时长 %.2f s' % (time.time() - start))


# 单次转换
def trans_stl_to_ply(path):
    pcd_ply = _trans(path)
    points, _ = get_mesh_data(pcd_ply, False)
    with open(r'D:/desktop/teeth_6_note/txt/PointCloud/Cloud2.txt', 'a+') as f:
        for elem1 in points.squeeze():
            f.write(str(elem1).strip('[').strip(']') + '\n')
    print('::转换成功')


if __name__ == '__main__':
    file_path = r'D:/desktop/Mesh.ply'
    trans_stl_to_ply(file_path)
