"""
測試自己的資料集，並存成檢測結果圖
"""


import torch, os, cv2
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch
import scipy.special, tqdm
import numpy as np
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor, tusimple_row_anchor

# 指定測試的配置訊息
backbone = "18" # 骨幹網路
dataset = "11" # 資料集類型
griding_num = 100 # 網格數
# test_model = "tusimple_18.pth" # 預訓練模型路徑 tusimple_18.pth
test_model = "放入欲訓練模型，自訓練or預訓練"
# data_root = "" # 開源資料集測試路徑
data_root = "自定義測試路徑" # 自定義測試路徑
data_save = "保存檢測結果" # 保存檢測結果



from PIL import Image
import os
import numpy as np
import glob



def loader_func(path):
    return Image.open(path)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, path, img_transform = None):
        super(TestDataset, self).__init__()
        self.path = path
        self.ing_transform = img_transform
        # self.list: 儲存測試圖片的相對路徑

    def __getitem__(self, index):
        name = glob.glob('%s/*.jpg'%self.path)[index]
        img = loader_func(name)

        if self.img_transform is not None:
            img = self.img_transform(img)
        return img, name

    def __len__(self):
        return len(self.list)

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True # 加速

    # args, cfg = merge_config() # 用終端機指定配置訊息
    dist_print('start testing...')
    assert backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    if dataset == 'CULane':
        cls_num_per_lane = 18
    elif dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
    #     raise NotImplementedError
        cls_num_per_lane = 56

    net = parsingNet(pretrained = False, backbone=backbone, cls_dim = (griding_num+1, cls_num_per_lane, 4),
                    use_aux=False).cuda() # we dont need auxiliary segmentation in testing

    state_dict = torch.load(test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()
    # 圖像格式統一: (288, 800), 圖像張量, 歸一化(標準化?)
    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    if dataset == 'CULane':
        splits = ['test0_normal.txt', 'test1_crowd.txt', 'test2_hlight.txt', 'test3_shadow.txt', 'test4_noline.txt', 'test5_arrow.txt', 'test6_curve.txt', 'test7_cross.txt', 'test8_night.txt']
        datasets = [LaneTestDataset(data_root, os.path.join(data_root, 'list/test_split/'+split),img_transform = img_transforms) for split in splits]
        
        img_w, img_h = 1640, 590
        row_anchor = culane_row_anchor
    elif dataset == 'Tusimple':
        splits = ['test.txt']
        datasets = [LaneTestDataset(data_root,os.path.join(data_root, split),img_transform = img_transforms) for split in splits]
        img_w, img_h = 1280, 720
        row_anchor = tusimple_row_anchor
    else: # 自定義資料集
        # raise NotImplementedError
        datasets = TestDataset(data_root, img_transform = img_transforms)
        img_w, img_h = 1280, 720
        row_anchor = tusimple_row_anchor

    for dataset in zip(datasets): # split: 圖片列表 datasets: 統一格式之後的資料集
        loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = 1) # 載入資料集
        # fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        # print(split[:-3] + "avi")
        # vout = cv2.VideoWriter(split[:-3] + "avi", fourcc, 30.0, (img_w, img_h)) # 保存結果成影片
        for i, data in enumerate(tqdm.tqdm(loader)): # 進度條顯示進度
            imgs, names = data # imgs: 圖像張量，圖像相對路徑
            imgs = imgs.cuda() # 使用GPU
            with torch.no_grad(): # 測試碼不計算梯度
                out = net(imgs) # 模型預測 輸出張量: [1,101,56,4]

            col_sample = np.linspace(0, 800 - 1, griding_num)
            col_sample_w = col_sample[1] - col_sample[0]


            out_j = out[0].data.cpu().numpy() # 數據類型轉換成numpy [101, 56, 4]
            out_j = out_j[:, ::-1, :] # 將第二維度倒著取[101, 56, 4]
            prob = scipy.special.softmax(out_j[:-1, :, :], axis=0) # [100, 56, 4]softmax計算(機率標準化到0-1之間且沿著維度0機率總和=1)
            idx = np.arange(griding_num) + 1 # 產生0-100
            idx = idx.reshape(-1, 1, 1) # [100, 1, 1]
            loc = np.sum(prob * idx, axis=0) # [56, 4]
            out_j = np.argmax(out_j, axis=0) # 返回最大值的索引
            loc[out_j == griding_num] = 0 # 若最大值的索引 = griding_num, 則歸零
            out_j = loc # [56, 4]

            # import pdb; pdb.set_trace()
            vis = cv2.imread(os.path.join(data_root,names[0])) # 讀取圖像[720, 1280, 3]
            for i in range(out_j.shape[1]): # 走遍所有列
                if np.sum(out_j[:, i] != 0) > 2: # 非0單位的格數<2
                    sum1 = np.sum(out_j[:, i] != 0)
                    for k in range(out_j.shape[0]): # 走遍所有行
                        if out_j[k, i] > 0:
                            ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                            cv2.circle(vis,ppp,5,(0,255,0),-1)
            # 保存檢測結果圖
            cv2.imwrite(os.path.join(data_save, os.path.basename(names[0])), vis)
        
        # 保存影片結果
        #     vout.write(vis)
        
        # vout.release()