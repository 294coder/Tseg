from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from PointNetPartSeg import load_module_opti
from processing_data import convert_to_datasets
from processing_data import module_predict_visualization

path = r'D:/desktop/1'
module_params_path = r'D:\codeblock\code\dachuang\logs\pointnet_teeth.pth'
if __name__ == "__main__":
    module, _ = load_module_opti(module_params_path)
    module.to('cuda')
    datasets = convert_to_datasets(path, predict=True)
    dataloader = DataLoader(datasets, 2)
    for X, y in dataloader:
        X = X.transpose(1, 2).float().cuda()

        seg_pred, trans_pred = module(X, for_teeth=True)
        seg_pred = seg_pred.contiguous().view(-1, 5)  # [bs*num_pts, 5]
        pred_choice = seg_pred.data.max(1)[1]

        X = X.transpose(2, 1)
        for i in range(len(X)):
            p = X[i]
            left = i * 8192
            right = (i + 1) * 8192
            pred = pred_choice[left:right]
            acc = accuracy_score(y[i].detach().cpu().numpy(), pred.detach().cpu().numpy())
            print(acc)
            module_predict_visualization(p, pred)
