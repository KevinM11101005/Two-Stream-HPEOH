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
from common.utils.transforms import ScaleNorm
from common.codecs.simcc_label import SimCCLabel
from common.codecs.keypoint_eval import simcc_pck_accuracy
from mmpose.utils.tensor_utils import to_numpy
from mmcv.cnn import ConvModule
from main.config import cfg
import numpy as np
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self, backbone_fpn, block_1, FIT, crossvit, SET, regressor, final_layer, mlp, gau, cls_x, cls_y, decoder, kldloss):
        super(Model, self).__init__()
        self.backbone_fpn = backbone_fpn
        self.FIT = FIT
        self.block_1 = block_1
        self.crossvit = crossvit
        self.SET = SET
        self.regressor = regressor
        self.final_layer = final_layer
        self.mlp = mlp
        self.gau = gau
        self.cls_x = cls_x
        self.cls_y = cls_y
        self.decoder = decoder
        self.kldloss = kldloss
    
    def forward(self, inputs, targets, UX_out, mode):
        p_feats, s_feats = self.backbone_fpn(inputs['img']) # inputs['img']
        b_feats = self.block_1(UX_out['skeleton_map'])

        feats = self.crossvit(p_feats, b_feats)

        if cfg.SET:
            feats = self.SET(feats, feats)

        if cfg.simcc:
            feats_t = self.final_layer(feats)
            feats_t = torch.flatten(feats_t, 2)
            feats_t = self.mlp(feats_t)
            feats_g = self.gau(feats_t)
            pred_x = self.cls_x(feats_g)
            pred_y = self.cls_y(feats_g)
        else:
            gt_mano_params = None
            out_hm, encoding, preds_joints_img = self.regressor(feats, gt_mano_params)

        if mode == 'train':
           # loss functions
            loss = {}
            if cfg.simcc:
                encode_joints = self.decoder.encode(targets['joints_img'].cpu().numpy())
                pred_simcc = (pred_x, pred_y)
                gt_simcc = (torch.from_numpy(encode_joints['keypoint_x_labels']).cuda(), torch.from_numpy(encode_joints['keypoint_y_labels']).cuda())
                loss['kpt'] = self.kldloss(pred_simcc, gt_simcc, torch.from_numpy(encode_joints['keypoint_weights']).cuda())
                # calculate accuracy
                acc = {}
                _, avg_acc, _ = simcc_pck_accuracy(
                    output=to_numpy(pred_simcc),
                    target=to_numpy(gt_simcc),
                    simcc_split_ratio=cfg.codec['simcc_split_ratio'],
                    mask=encode_joints['keypoint_weights'] > 0,
                )
                avg_hand = torch.tensor(avg_acc, dtype=torch.float32).cuda()
                acc['pck_hand'] = avg_hand
                return loss, acc
            else:
                loss['joints_img'] = cfg.lambda_joints_img * F.mse_loss(preds_joints_img[0], targets['joints_img'])
                return loss
        else:
            # test output
            out = {}
            if cfg.simcc:
                keypoints, scores = self.decoder.decode(to_numpy(pred_x), to_numpy(pred_y))
                out['keypoints'] = keypoints
                out['keypoint_scores'] = scores
                return out
            else:
                out['joints_coord_img'] = preds_joints_img[0]
                return out
    
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

def get_HON_model(mode):
    backbone_fpn = FPN(pretrained=True)
    FIT = Transformer(injection=True, depth=1) # feature injecting transformer
    crossvit = CrossAttention(256, 256, depth=1)
    SET = Transformer(injection=False) # self enhancing transformer
    regressor = Regressor()
    block_1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
    final_layer = ConvModule(
                    cfg.in_channels,
                    cfg.out_channels,
                    kernel_size=cfg.final_layer_kernel_size,
                    stride=1,
                    padding=cfg.final_layer_kernel_size // 2,
                    norm_cfg=dict(type='BN', requires_grad=True),
                    act_cfg=dict(type='ReLU'))
    mlp = nn.Sequential(
            ScaleNorm(cfg.flatten_dims),
            nn.Linear(cfg.flatten_dims, cfg.gau_cfg['hidden_dims'], bias=False))
    gau = RTMCCBlock(
            cfg.out_channels,
            cfg.gau_cfg['hidden_dims'],
            cfg.gau_cfg['hidden_dims'],
            s=cfg.gau_cfg['s'],
            expansion_factor=cfg.gau_cfg['expansion_factor'],
            dropout_rate=cfg.gau_cfg['dropout_rate'],
            drop_path=cfg.gau_cfg['drop_path'],
            attn_type='self-attn',
            act_fn=cfg.gau_cfg['act_fn'],
            use_rel_bias=cfg.gau_cfg['use_rel_bias'],
            pos_enc=cfg.gau_cfg['pos_enc'])
    cls_x = nn.Linear(cfg.gau_cfg['hidden_dims'], cfg.W, bias=False)
    cls_y = nn.Linear(cfg.gau_cfg['hidden_dims'], cfg.H, bias=False)
    decoder = SimCCLabel(input_size=cfg.codec['input_size'],
                sigma=cfg.codec['sigma'],
                simcc_split_ratio=cfg.codec['simcc_split_ratio'],
                normalize=cfg.codec['normalize'],
                use_dark=cfg.codec['use_dark'])
    kldloss = KLDiscretLoss(use_target_weight=cfg.loss['use_target_weight'],
                beta=cfg.loss['beta'],            
                label_softmax=cfg.loss['label_softmax'])
    
    if mode == 'train':
        cls_x.apply(init_weights)
        cls_y.apply(init_weights)
        
    model = Model(backbone_fpn, block_1, FIT, crossvit, SET, regressor, final_layer, mlp, gau, cls_x, cls_y, decoder, kldloss)
    
    return model