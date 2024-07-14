import sys
import glob
import os
import os.path as osp
import argparse
import json
import numpy as np
import cv2
import torch
import tkinter as tk
from tkinter import filedialog
current_path = os.getcwd()
sys.path.append(current_path)
from PIL import Image
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from main.config import cfg
# dynamic model import
if cfg.backbone == 'fpn':
    from main.model import get_model
elif cfg.backbone == 'unext': 
    from main.model_UX import get_UX_model
    from main.model_HON import get_HON_model

from common.utils.preprocessing import load_img, process_bbox, generate_patch_image
from common.utils.vis import draw_bbox, draw_skeletons, overlay_bbox_image
from torchinfo import summary

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')

    args = parser.parse_args()
    args.gpu_ids = '0'
    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

def choose_file():
    # 创建 Tkinter 根窗口
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口

    # 使用 filedialog 模块的 askopenfilename 函数来选择文件
    file_path = filedialog.askopenfilename()

    # 返回所选文件的路径
    return file_path

if __name__ == '__main__':
    # argument parsing
    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cudnn.benchmark = True
    transform = transforms.ToTensor()
    
    # hard coding
    #################################
    model_idx = 10
    #################################
    img_path = choose_file()
    print(img_path)
    bbox_ori = draw_bbox(img_path)#[340.8, 232.0, 20.7, 20.7] # xmin, ymin, width, height 
    print(f'bbox:{bbox_ori}')

    if cfg.backbone == 'unext':
        itr = 0
        # model snapshot load
        model_path = cfg.outputname+'/model_dump/snapshot_UX_{}.pth.tar'.format(model_idx)
        assert osp.exists(model_path), 'Cannot find model at ' + model_path
        print('Load checkpoint from {}'.format(model_path))
        ux_model = get_UX_model('test')

        ux_model = DataParallel(ux_model).cuda()
        ckpt = torch.load(model_path)
        ux_model.load_state_dict(ckpt['network'], strict=False)
        ux_model.eval()

        # model snapshot load
        model_path = cfg.outputname+'/model_dump/snapshot_HON_{}.pth.tar'.format(model_idx)
        assert osp.exists(model_path), 'Cannot find model at ' + model_path
        print('Load checkpoint from {}'.format(model_path))
        hon_model = get_HON_model('test')

        hon_model = DataParallel(hon_model).cuda()
        ckpt = torch.load(model_path)
        hon_model.load_state_dict(ckpt['network'], strict=False)
        hon_model.eval()
    else:
        # model snapshot load
        model_path = cfg.outputname+'/model_dump/snapshot_{}.pth.tar'.format(model_idx)
        
        assert osp.exists(model_path), 'Cannot find model at ' + model_path
        print('Load checkpoint from {}'.format(model_path))
        model = get_model('test')

        model = DataParallel(model).cuda()
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'], strict=False)
        model.eval()

    # prepare input image
    transform = transforms.ToTensor()
    ori_img = cv2.imread(img_path)
    original_img = load_img(img_path)
    original_img_height, original_img_width = original_img.shape[:2]
    # prepare bbox
    bbox = process_bbox(bbox_ori, original_img_width, original_img_height)
    img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape) 
    img = transform(img.astype(np.float32))/255
    img = img.cuda()[None,:,:,:]

    # forward pass to the model
    inputs = {'img': img} # cfg.input_img_shape[1], cfg.input_img_shape[0], 3
    targets = {}
    
    if cfg.backbone == 'unext':
        with torch.no_grad():
            ux_out = ux_model(inputs, targets, 'test', itr)
            out = hon_model(inputs, targets, ux_out, 'test')
    else:
        with torch.no_grad():
            out = model(inputs, targets, 'test')
    img = (img[0].cpu().numpy().transpose(1, 2, 0)*255).astype(np.uint8) # 
    if cfg.simcc:
        # get hand keypoints and keypoint_scores
        keypoints, keypoint_scores = out['keypoints'], out['keypoint_scores']
        keypoints_restored = np.dot(bb2img_trans, np.concatenate((keypoints[0], np.ones((keypoints[0].shape[0], 1))), axis=1).transpose(1, 0))
        keypoints_restored = keypoints_restored[:2, :].transpose(1, 0)
        final_image = draw_skeletons(ori_img, np.expand_dims(keypoints_restored,axis=0))

        bbox_image = draw_skeletons(img, out['keypoints'])
    else:
        keypoints = out['joints_coord_img'].cpu().numpy()
        keypoints[:,:,0] *= cfg.input_img_shape[1]
        keypoints[:,:,1] *= cfg.input_img_shape[0]
        keypoints_restored = np.dot(bb2img_trans, np.concatenate((keypoints[0], np.ones((keypoints[0].shape[0], 1))), axis=1).transpose(1, 0))
        keypoints_restored = keypoints_restored[:2, :].transpose(1, 0)
        
        final_image = draw_skeletons(ori_img, np.expand_dims(keypoints_restored,axis=0))

        bbox_image = draw_skeletons(img, keypoints)
    restored_img = cv2.warpAffine(bbox_image, bb2img_trans[:2], (original_img_width, original_img_height))
    restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
    # final_image = overlay_bbox_image(ori_img, restored_img, bbox)
    print(f're_bbox: {bbox}')
    print(f'restored_img shape: {restored_img.shape}')
    
    imgs = np.hstack([restored_img, final_image])
    cv2.imshow('left: restored_img, right: final_image', imgs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()