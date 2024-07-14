# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

class KLDiscretLoss(nn.Module):
    """Discrete KL Divergence loss for SimCC with Gaussian Label Smoothing.
    Modified from `the official implementation.

    <https://github.com/leeyegy/SimCC>`_.
    Args:
        beta (float): Temperature factor of Softmax. Default: 1.0.
        label_softmax (bool): Whether to use Softmax on labels.
            Default: False.
        label_beta (float): Temperature factor of Softmax on labels.
            Default: 1.0.
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
        mask (list[int]): Index of masked keypoints.
        mask_weight (float): Weight of masked keypoints. Default: 1.0.
    """

    def __init__(self,
                 beta=1.0,
                 label_softmax=False,
                 label_beta=10.0,
                 use_target_weight=True,
                 mask=None,
                 mask_weight=1.0):
        super(KLDiscretLoss, self).__init__()
        self.beta = beta
        self.label_softmax = label_softmax
        self.label_beta = label_beta
        self.use_target_weight = use_target_weight
        self.mask = mask
        self.mask_weight = mask_weight

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def criterion(self, dec_outs, labels):
        """Criterion function."""
        log_pt = self.log_softmax(dec_outs * self.beta)
        if self.label_softmax:
            labels = F.softmax(labels * self.label_beta, dim=1)
        loss = torch.mean(self.kl_loss(log_pt, labels), dim=1)
        return loss

    def forward(self, pred_simcc, gt_simcc, target_weight):
        """Forward function.

        Args:
            pred_simcc (Tuple[Tensor, Tensor]): Predicted SimCC vectors of
                x-axis and y-axis.
            gt_simcc (Tuple[Tensor, Tensor]): Target representations.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.
        """
        N, K, _ = pred_simcc[0].shape
        loss = 0

        if self.use_target_weight:
            weight = target_weight.reshape(-1)
        else:
            weight = 1.

        for pred, target in zip(pred_simcc, gt_simcc):
            pred = pred.reshape(-1, pred.size(-1))
            target = target.reshape(-1, target.size(-1))

            t_loss = self.criterion(pred, target).mul(weight)

            if self.mask is not None:
                t_loss = t_loss.reshape(N, K)
                t_loss[:, self.mask] = t_loss[:, self.mask] * self.mask_weight

            loss = loss + t_loss.sum()

        return loss / K