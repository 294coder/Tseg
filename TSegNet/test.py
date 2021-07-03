from Three_registration import process_icp_usual
from utils.utils import make_centroids, centroids_clustering, upsample_teeth
from utils.open3d_utils import *
from pct_partseg import Point_Transformer_partseg
from processing_data import density_adaptive_downsampling
from sklearn.preprocessing import StandardScaler
import time
import os


# source_path = r'D:\desktop\teeth_6_note\001-050牙体完整数据牙齿分割（txt格式）\1-7\005-lu\all.txt'  # 缺损
# target_path = r"D:\desktop\teeth_6_note\001-050牙体完整数据牙齿分割（txt格式）\1-7\006-lu\all.txt"  # 模板
#
# s = np.loadtxt(source_path)
# s = array_to_pointcloud(s)
# t = array_to_pointcloud(np.loadtxt(target_path))
#
# # icp registration
# matrix = process_icp_usual(source_path, target_path)
#
# s.transform(matrix)
# s_pts = np.asarray(s.points)
#
# # four parts
# t1 = np.loadtxt(r"D:\desktop\teeth_6_note\001-050牙体完整数据牙齿分割（txt格式）\1-7\005-lu\24.txt")
# t2 = np.loadtxt(r"D:\desktop\teeth_6_note\001-050牙体完整数据牙齿分割（txt格式）\1-7\005-lu\25.txt")
# t3 = np.loadtxt(r"D:\desktop\teeth_6_note\001-050牙体完整数据牙齿分割（txt格式）\1-7\005-lu\26.txt")
# t4 = np.loadtxt(r"D:\desktop\teeth_6_note\001-050牙体完整数据牙齿分割（txt格式）\1-7\005-lu\27.txt")
#
# centroids = make_centroids(*[t1, t2, t3, t4])
#
# # clustering
# clusters = []
# for c in centroids:  # clustering
#     clust = centroids_clustering(c, s_pts, factors=0.29)
#     clust = array_to_pointcloud(clust).paint_uniform_color([0.5, 0.5, 0.5])
#     clusters.append(clust)
#
# clusters.append(s)
# o3d.visualization.draw_geometries(clusters)
#
# cluster_pts = [np.asarray(cl.points) for cl in clusters]
# pts = cluster_pts[0]
# for i in range(1, len(cluster_pts) - 1):
#     pts_cat = np.concatenate([pts, cluster_pts[i]], axis=0)
#
# pcd = array_to_pointcloud(pts_cat)
# o3d.visualization.draw_geometries([pcd])
#
# # 2 class
# module2 = Point_Transformer_partseg(part_num=2)
# read_path = r'D:\codeblock\code\dachuang\logs\pct_2class5.pth'
# module2.load_state_dict(torch.load(read_path, map_location='cpu')['module_params'])
# print('load module successfully')
#
# # 4 class
# module4 = Point_Transformer_partseg(part_num=4).cuda()
# read_path = r'D:\codeblock\code\dachuang\logs\PCT_weight_2021-05-10-19.pth'
# model_dict = torch.load(read_path, map_location='cpu')['module_params']
# module4.load_state_dict(model_dict)
# module4.cpu()
# print('module load success')
#
# # rename
# from sklearn.preprocessing import StandardScaler
#
# # downsample
# # X2 is before scale
# torch.manual_seed(2021)
# X = density_adaptive_downsampling(pts_cat)
# pcd = array_to_pointcloud(X)
# o3d.visualization.draw_geometries([pcd])
# X2 = torch.tensor(X)
#
# # scale
# # X is after scale
# X = torch.tensor(StandardScaler().fit_transform(X)[None, :, :])
# X = X.transpose(1, 2).float()
#
# # 2 class predict
# module2.eval()
# seg_pred2 = module2(X).transpose(1, 2)
# seg_pred2 = seg_pred2.contiguous().view(-1, 2)
# pred_choice2 = seg_pred2.data.max(1)[1]
# mask = pred_choice2 == 1
# mask2 = pred_choice2 == 0
# pred_pts = X.squeeze().transpose(0, 1)[mask]
#
# X2_fg = X2[mask]
# X2_bg = X2[mask2]
# pts_bg_pcd = array_to_pointcloud(X2_bg.numpy()).paint_uniform_color([0, 1, 0])
# pts_2class_pcd = array_to_pointcloud(X2_fg.numpy()).paint_uniform_color([0, 0, 1])
# o3d.visualization.draw_geometries([pts_2class_pcd,pts_bg_pcd])
# X2 = X2_fg.unsqueeze(0)
#
# # 4 class predict
# module4.eval()
# X = pred_pts.unsqueeze(0).transpose(1, 2)
# seg_pred4 = module4(X).transpose(1, 2)
# seg_pred4 = seg_pred4.contiguous().view(-1, 4)  # [bs*num_pts, 5]
# pred_choice = seg_pred4.data.max(1)[1]
# pred_choice = pred_choice.contiguous().view(1, -1)
# for i, j in zip(X.transpose(1, 2), pred_choice):
#     bounding_box_visualization(i, j)


def predict_all_file(func):
    def wrapper(all_file_path, **kwargs):
        print("程序运行")
        t1 = time.time()
        dirlst = os.listdir(all_file_path)
        for p in dirlst:
            p = os.path.join(all_file_path, p)
            func(path=p, **kwargs)
        t2 = time.time()
        print("程序运行时间 %.4f s" % (t2 - t1))

    return wrapper


@predict_all_file
def module_predict(path, module2class_path, module4class_path):
    # 2 class
    module2 = Point_Transformer_partseg(part_num=2).cuda()
    module2.load_state_dict(torch.load(module2class_path, map_location='cpu')['module_params'])
    module2.eval()

    # 4 class
    module4 = Point_Transformer_partseg(part_num=4).cuda()
    model_dict = torch.load(module4class_path, map_location='cpu')['module_params']
    module4.load_state_dict(model_dict)
    module4.eval()

    teeth_np = np.loadtxt(path)[:, :3]
    teeth_np = density_adaptive_downsampling(teeth_np, size=8192)
    np.random.shuffle(teeth_np)
    X = teeth_np[:8192]
    # pcd = array_to_pointcloud(X)
    # o3d.visualization.draw_geometries([pcd])
    std_teeth = StandardScaler().fit_transform(X)
    # pcd = array_to_pointcloud(std_teeth)
    # o3d.visualization.draw_geometries([pcd])
    X2 = torch.tensor(std_teeth).unsqueeze(0).transpose(1, 2).float().cuda()

    seg_pred2 = module2(X2).transpose(1, 2)
    seg_pred2 = seg_pred2.contiguous().view(-1, 2)
    pred_choice2 = seg_pred2.data.max(1)[1]
    mask = pred_choice2 == 1
    mask2 = pred_choice2 == 0
    pred_pts = X2.squeeze().transpose(0, 1)[mask]
    pred_bg = X2.squeeze().transpose(0, 1)[mask2]
    pcd_np = pred_pts.cpu().numpy()
    pcdbg_np = pred_bg.cpu().numpy()
    # np.savetxt('D:/desktop/1.txt', pcdbg_np, fmt="%.4f")
    # np.savetxt('D:/desktop/2.txt', pcd_np, fmt="%.4f")
    pcd = array_to_pointcloud(pcd_np).paint_uniform_color([0, 0, 1])
    pcd_bg = array_to_pointcloud(pcdbg_np).paint_uniform_color([.5, .5, .5])
    o3d.visualization.draw_geometries([pcd, pcd_bg])

    # std_teeth = StandardScaler().fit_transform(pcd_np)
    pred_pts = upsample_teeth(pred_pts)
    # np.savetxt('D:/desktop/output.txt',pred_pts)
    pred_pts = torch.from_numpy(StandardScaler().fit_transform(pred_pts.numpy()))
    X = pred_pts.unsqueeze(0).transpose(1, 2).cuda()
    # X = X2
    seg_pred4 = module4(X).transpose(1, 2)
    seg_pred4 = seg_pred4.contiguous().view(-1, 4)  # [bs*num_pts, 5]
    pred_choice = seg_pred4.data.max(1)[1]
    # pred_choice = pred_choice.contiguous().view(1, -1)
    # for i, j in zip(X.transpose(1, 2).cpu(), pred_choice):
    #     bounding_box_visualization(i, j, for_4class=True)

    X = X.transpose(1, 2).cpu().squeeze().numpy()
    pred_choice=pred_choice.cpu().numpy()
    color_all = [[.5, .5, .5], [0, 0, 1], [0, 1, 0], [1, 0, 0], [.2, .4, .8], [1, 1, .3],
                 [.2, .7, 1], [.5, 1, 1], [.4, .5, .6]]

    bg = array_to_pointcloud(X[np.argwhere(pred_choice == 0)].squeeze())  # background
    t1 = array_to_pointcloud(X[np.argwhere(pred_choice == 1)].squeeze())
    t2 = array_to_pointcloud(X[np.argwhere(pred_choice == 2)].squeeze())
    t3 = array_to_pointcloud(X[np.argwhere(pred_choice == 3)].squeeze())
    t2.paint_uniform_color(color_all[2])
    t3.paint_uniform_color(color_all[3])
    # uniform color
    bg.paint_uniform_color(color_all[0])
    t1.paint_uniform_color(color_all[1])
    o3d.visualization.draw_geometries([bg, t1, t2, t3])
    return seg_pred4, pred_choice


if __name__ == '__main__':
    module2class_path = r'D:\codeblock\code\dachuang\logs\pct_2class6-test.pth'
    module4class_path = r'D:\codeblock\code\dachuang\logs\PCT_weight_2021-06-20-00.pth'
    path = r'D:/desktop/main_read'
    module_predict(path, module2class_path=module2class_path, module4class_path=module4class_path)
