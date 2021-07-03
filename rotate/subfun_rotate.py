import open3d as o3d
import numpy as np
import math
from tqdm import trange
import pickle

PI = math.pi
cos = math.cos
sin = math.sin


def impliment_o3d(m: np.ndarray):
    l1 = np.append(m[0], 0)[np.newaxis, :]
    l2 = np.append(m[1], 0)[np.newaxis, :]
    l3 = np.append(m[2], 0)[np.newaxis, :]
    l4 = np.array([[0,0,0,1]])
    new_m = np.concatenate([l1, l2, l3,l4],axis=0)
    return new_m


def rotate(X, A=None):
    """
    :type X: np.ndarray
    :type A: np.ndarray

    :param X: Input teeth data which shape is [N,3]
    :param A: rotate minimum angle
    :return: a dict within X: transformed matrix from original data
                           A: rotate angles
                           M: rotate matrix which can be used in package: open3d function named open3d_class.transform()
    """

    def _define_theta(theta_x=0, theta_y=0, theta_z=0):

        R_x = np.array([
            [1, 0, 0],
            [0, cos(theta_x), -sin(theta_x)],
            [0, sin(theta_x), cos(theta_x)]
        ])
        R_y = np.array([
            [cos(theta_y), 0, sin(theta_y)],
            [0, 1, 0],
            [-sin(theta_y), 0, cos(theta_y)]
        ])
        R_z = np.array([
            [cos(theta_z), -sin(theta_z), 0],
            [sin(theta_z), cos(theta_z), 0],
            [0, 0, 1]
        ])

        return R_x, R_y, R_z

    if A != None:
        angle = A
    else:
        angle = np.array(range(-180, 180, 5)) * PI / 180

    N = len(angle)
    Max_S = 0

    for i1 in trange(0, N):
        # print('i1 is ', i1)
        for i2 in range(0, N):
            # print('i2 is', i2)
            for i3 in range(0, N):
                theta_x = angle[i1]
                theta_y = angle[i2]
                theta_z = angle[i3]

                R_x, R_y, R_z = _define_theta(theta_x, theta_y, theta_z)
                temp_M = R_x @ R_y @ R_z
                temp_X = temp_M @ X.T
                temp_X = temp_X.T

                minmax_x = [min(temp_X[:, 0]), max(temp_X[:, 0])]
                minmax_y = [min(temp_X[:, 1]), max(temp_X[:, 1])]
                s = (minmax_x[1] - minmax_x[0]) * (minmax_y[1] - minmax_y[0])
                if s > Max_S:
                    Max_S = s
                    Rotate = {
                        'A': np.array([theta_x, theta_y, theta_z]),
                        'X': temp_X,
                        'M': temp_M
                    }
    theta_y = 45 * PI / 180
    theta_z = 30 * PI / 180
    _, R_y, R_z = _define_theta(theta_x, theta_y, theta_z)
    Rotate['X'] = (R_y @ R_z @ Rotate['X'].T).T
    Rotate['A'] = Rotate['A'] + np.array([0, theta_y, theta_z])
    Rotate['M'] = R_y @ R_z @ Rotate['M']

    return Rotate


def load_binary(path):
    f = open(path, 'rb')
    s2 = f.readlines()
    return pickle.loads(s2[0])


def save_rotate_txt(s: str, path: str) -> bool:
    assert s is not None
    assert path is not None

    f = open(path, 'wb')
    f.write(s)
    return True


if __name__ == '__main__':
    from load_data import get_noted_teeth_data as gntd
    from load_data import array_to_pointcloud
    import torch

    m = np.random.randn(3, 3)
    SAVE_PATH = r'D:\codeblock\code\dachuang\TeethTemplate\matrix.txt'
    PATH = r"D:\codeblock\code\dachuang\TeethTemplate\result_icp_ptp.ply"

    d={
        'A':123,
        'B':'this is a dict'
    }
    save_rotate_txt(pickle.dumps(d),SAVE_PATH)
    print(load_binary(SAVE_PATH))


    # pcd = o3d.io.read_point_cloud(PATH)
    # print('::before transform')
    # o3d.visualization.draw_geometries([pcd])
    # data = np.asarray(pcd.points)
    # data_x=data[:,0]
    # data_y=data[:,1]
    # import pandas as pd
    # df=pd.DataFrame(data=data,columns=['x','y','z'])
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # # sns.set_theme(context='talk')
    # # sns.jointplot(x='x',y='y',data=df)
    # plt.plot(data_x,data_y,'.',markersize=.2)
    # plt.show()
    # m = np.asarray([[-0.97905317, 0.18619948, - 0.08236898, 0],
    #                 [0.20078367, 0.95006366, - 0.23888274, 0],
    #                 [0.03377593, - 0.25041725, - 0.96754865, 0],
    #                 [0, 0, 0, 1]])
    # pcd = pcd.transform(m)
    # print('::after transform')
    # o3d.visualization.draw_geometries([pcd])
    #
    # data = np.asarray(pcd.points)
    # data_x=data[:,0]
    # data_y=data[:,1]
    # import pandas as pd
    # df=pd.DataFrame(data=data,columns=['x','y','z'])
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # # sns.set_theme(context='talk')
    # # sns.jointplot(x='x',y='y',data=df)
    # plt.plot(data_x,data_y,'.',markersize=.2)
    # plt.show()
