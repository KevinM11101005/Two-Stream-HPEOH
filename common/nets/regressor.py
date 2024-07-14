import torch
import torch.nn as nn
from torch.nn import functional as F
from common.nets.hand_head import hand_regHead

class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.hand_regHead = hand_regHead()
    
    def forward(self, feats, gt_mano_params=None):
        out_hm, encoding, preds_joints_img = self.hand_regHead(feats)

        return out_hm, encoding, preds_joints_img
