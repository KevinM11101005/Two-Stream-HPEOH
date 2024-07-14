import torch
import numpy as np
from main.config import cfg
import torch.nn as nn
from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.utils import digit_version
from scipy.spatial import procrustes

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:,0] / cam_coord[:,2] * f[0] + c[0]
    y = cam_coord[:,1] / cam_coord[:,2] * f[1] + c[1]
    z = cam_coord[:,2]
    return np.stack((x,y,z),1)

def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:,0] - c[0]) / f[0] * pixel_coord[:,2]
    y = (pixel_coord[:,1] - c[1]) / f[1] * pixel_coord[:,2]
    z = pixel_coord[:,2]
    return np.stack((x,y,z),1)

def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord

def cam2world(cam_coord, R, t):
    world_coord = np.dot(np.linalg.inv(R), (cam_coord - t.reshape(1,3)).transpose(1,0)).transpose(1,0)
    return world_coord

def rigid_transform_3D(A, B):
    n, dim = A.shape
    centroid_A = np.mean(A, axis = 0)
    centroid_B = np.mean(B, axis = 0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))

    varP = np.var(A, axis=0).sum()
    c = 1/varP * np.sum(s) 

    t = -np.dot(c*R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t

def rigid_align(A, B):
    c, R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(c*R, np.transpose(A))) + t
    return A2

def transform_joint_to_other_db(src_joint, src_name, dst_name):
    src_joint_num = len(src_name)
    dst_joint_num = len(dst_name)

    new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]), dtype=np.float32)
    for src_idx in range(len(src_name)):
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_joint[dst_idx] = src_joint[src_idx]

    return new_joint

def compute_mpjpe(predicted, ground_truth):
    """
    Computes Mean Per Joint Position Error (MPJPE).
    
    Args:
    predicted (np.ndarray): Predicted joint positions of shape (J, 2),
                            J is the number of joints.
    ground_truth (np.ndarray): Ground truth joint positions of shape (J, 2).

    Returns:
    float: MPJPE value.
    """
    assert predicted.shape == ground_truth.shape, "Shape of predicted and ground truth must be the same"

    # Calculate the Euclidean distance between predicted and ground truth joints
    error = np.linalg.norm(predicted - ground_truth, axis=-1)
    
    # Calculate mean error per joint
    mpjpe = np.mean(error)
    
    return mpjpe

def compute_pa_mpjpe(predicted, ground_truth):
    """
    Computes Procrustes Aligned Mean Per Joint Position Error (PA MPJPE).
    
    Args:
    predicted (np.ndarray): Predicted joint positions of shape (J, 2),
                            J is the number of joints.
    ground_truth (np.ndarray): Ground truth joint positions of shape (J, 2).

    Returns:
    float: PA MPJPE value.
    """
    assert predicted.shape == ground_truth.shape, "Shape of predicted and ground truth must be the same"

    J, _ = predicted.shape
    total_error = 0.0
    
    # Procrustes analysis to align predicted with ground truth
    mtx1, mtx2, disparity = procrustes(ground_truth, predicted)
    
    # Calculate the Euclidean distance between aligned predicted and ground truth joints
    error = np.linalg.norm(mtx1 - mtx2, axis=-1)
    total_error += np.mean(error)
    
    # Calculate mean error per joint across all samples
    pa_mpjpe = total_error
    
    return pa_mpjpe

class ScaleNorm(nn.Module):
    """Scale Norm.

    Args:
        dim (int): The dimension of the scale vector.
        eps (float, optional): The minimum value in clamp. Defaults to 1e-5.

    Reference:
        `Transformers without Tears: Improving the Normalization
        of Self-Attention <https://arxiv.org/abs/1910.05895>`_
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The tensor after applying scale norm.
        """

        if torch.onnx.is_in_onnx_export() and \
                digit_version(TORCH_VERSION) >= digit_version('1.12'):

            norm = torch.linalg.norm(x, dim=-1, keepdim=True)

        else:
            norm = torch.norm(x, dim=-1, keepdim=True)
        norm = norm * self.scale
        return x / norm.clamp(min=self.eps) * self.g