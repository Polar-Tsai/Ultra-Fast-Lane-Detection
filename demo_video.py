"""

測試自己的影片，有機會進行real-time


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


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    dist_print('start testing...')
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    net = parsingNet(pretrained = False, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cls_num_per_lane,4),
                    use_aux=False).cuda() # we dont need auxiliary segmentation in testing

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([

        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


# =====新增程式開始=====
    cap = cv2.VideoCapture("****.avi") # 讀取影片文件，要確認攝影機的影片是否為avi檔
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    rval, frame = cap.read()
    frame = frame[490:1080, 0:1640, :]
    vout = cv2.VideoWriter("output.avi", fourcc, 30.0, (frame.shape[1], frame.shape[0]))
    print("w= {}, h = {}".format(cap.get(3), cap.get(4)))
    from PIL import Image

    print("CUDA running?:", torch.cuda.is_available())
    while 1:
        rval, frame = cap.read()
        if rval == False:
            break
        frame = frame[490:1080, 0:1640, :]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_ = Image.fromarray(img)
        imgs = img_transforms(img_)
        imgs = imgs.unsqueeze(0)
        imgs = imgs.cuda()
        with torch.no_grad():
            out = net(imgs)
        
        col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]

        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(cfg.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == cfg.griding_num] = 0
        out_j = loc

        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (
                        int(out_j[k, i] * col_sample_w * frame.shape[1] / 800) - 1, int(frame.shape[0] - k * 20) - 1 )
                        cv2.circle(frame, ppp, 5, (0, 255, 0), -1)
        vout.write(frame)

    vout.release()       
#      ======增加程式結束======


    # if cfg.dataset == 'CULane':
    #     splits = ['test0_normal.txt', 'test1_crowd.txt', 'test2_hlight.txt', 'test3_shadow.txt', 'test4_noline.txt', 'test5_arrow.txt', 'test6_curve.txt', 'test7_cross.txt', 'test8_night.txt']
    #     datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root, 'list/test_split/'+split),img_transform = img_transforms) for split in splits]
    #     img_w, img_h = 1640, 590
    #     row_anchor = culane_row_anchor
    # elif cfg.dataset == 'Tusimple':
    #     splits = ['test.txt']
    #     datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root, split),img_transform = img_transforms) for split in splits]
    #     img_w, img_h = 1280, 720
    #     row_anchor = tusimple_row_anchor
    # else:
    #     raise NotImplementedError
    # for split, dataset in zip(splits, datasets):
    #     loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle = False, num_workers=1)
    #     fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #     print(split[:-3]+'avi')
    #     vout = cv2.VideoWriter(split[:-3]+'avi', fourcc , 30.0, (img_w, img_h))
    #     for i, data in enumerate(tqdm.tqdm(loader)):
    #         imgs, names = data
    #         imgs = imgs.cuda()
    #         with torch.no_grad():
    #             out = net(imgs)

    #         col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
    #         col_sample_w = col_sample[1] - col_sample[0]


    #         out_j = out[0].data.cpu().numpy()
    #         out_j = out_j[:, ::-1, :]
    #         prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
    #         idx = np.arange(cfg.griding_num) + 1
    #         idx = idx.reshape(-1, 1, 1)
    #         loc = np.sum(prob * idx, axis=0)
    #         out_j = np.argmax(out_j, axis=0)
    #         loc[out_j == cfg.griding_num] = 0
    #         out_j = loc

    #         # import pdb; pdb.set_trace()
    #         vis = cv2.imread(os.path.join(cfg.data_root,names[0]))
    #         for i in range(out_j.shape[1]):
    #             if np.sum(out_j[:, i] != 0) > 2:
    #                 for k in range(out_j.shape[0]):
    #                     if out_j[k, i] > 0:
    #                         ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
    #                         cv2.circle(vis,ppp,5,(0,255,0),-1)
    #         vout.write(vis)
        
    #     vout.release()