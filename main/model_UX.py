import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial
from common.codecs.KLDiscretLoss import KLDiscretLoss
from common.nets.backbone_FPN import FPN
from common.nets.backbone_UNext import UNext
from common.nets.crossvit import CrossAttention
from common.nets.transformer import Transformer
from common.nets.regressor import Regressor
from common.nets.gau import RTMCCBlock
from common.utils.losses import BCEDiceLoss
from common.codecs.simcc_label import SimCCLabel
from common.codecs.keypoint_eval import simcc_pck_accuracy
from mmpose.utils.tensor_utils import to_numpy
from mmcv.cnn import ConvModule
import matplotlib.pyplot as plt
import cv2 
from main.config import cfg

def dice_metric_loss(ground_truth, predictions, smooth=1e-6):
    ground_truth = ground_truth.to(torch.float32)
    predictions = predictions.to(torch.float32)
    ground_truth = torch.flatten(ground_truth)
    predictions = torch.flatten(predictions)
    intersection = torch.sum(predictions * ground_truth)
    union = torch.sum(predictions) + torch.sum(ground_truth)

    dice = (2. * intersection + smooth) / (union + smooth)

    return 1 - dice

class Model(nn.Module):
    def __init__(self, backbone_unext):
        super(Model, self).__init__()
        self.backbone_unext = backbone_unext
        
    def forward(self, inputs, targets, mode, itr):
        u_feats = self.backbone_unext(inputs['img']) # inputs['img']

        if mode == 'train':
            # loss functions
            loss = {} 
            pred_skeleton_map = torch.sigmoid(u_feats) ## turn to 0 ~ 1
            gt_skeleton_map = targets['skeleton_map'] ## must be 0. and 1.(float)
            
            # loss_function = BCEDiceLoss()
            loss['BCE'] = dice_metric_loss(pred_skeleton_map, gt_skeleton_map)
            # calculate accuracy
            acc = {}
            predictions = pred_skeleton_map.cpu().detach().numpy()
            gt = gt_skeleton_map.cpu().numpy()

            predictions = (predictions > 0.5).astype(float)
            gt = (gt > 0.5).astype(float)

            # path = cfg.vis_dir+'/'+str(itr)+'.png'
            # result = cv2.hconcat([predictions[0, 0]*255, gt[0, 0]*255])
            # cv2.imwrite(path, result)

            num_correct = (predictions==gt).sum().item() ## 0. False 1. True
            num_pixels = cfg.train_batch_size * cfg.input_img_shape[0] * cfg.input_img_shape[1]
            acc['batch_pixels'] = torch.tensor(num_correct / num_pixels)

            # outs
            outs = {}
            outs['skeleton_map'] = torch.sigmoid(u_feats)
            
            return loss, acc, outs
        else:
            # test output
            outs = {}
            outs['skeleton_map'] = torch.sigmoid(u_feats)
            return outs

    
def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias,0)

def get_UX_model(mode):
    backbone_unext = UNext(in_channels=3, num_classes=1)
        
    model = Model(backbone_unext)
    
    return model