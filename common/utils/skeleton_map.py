import cv2
import numpy as np
import torch
import sys
import os

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
# 将项目根目录添加到sys.path
sys.path.append(project_root)

from main.config import cfg

def skeleton_map_gray(image_size, joints):
    if not cfg.simcc:
        joints[:,0] *= image_size[0]
        joints[:,1] *= image_size[1]

    image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

    # 将骨架点位转为整数坐标

    joints = np.array(joints, dtype=np.int32)
    # 定义骨架连接（根据你具体的骨架定义，这里是一个示例）
    if cfg.trainset == 'DEX_YCB':
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]
    elif cfg.trainset == 'HO3D':
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 17),
            (0, 4), (4, 5), (5, 6), (6, 18),
            (0, 7), (7, 8), (8, 9), (9, 20),
            (0, 10), (10, 11), (11, 12), (12, 19),
            (0, 13), (13, 14), (14, 15), (15, 16)
        ]

    # 画骨架连接
    for (i, j) in connections:
        cv2.line(image, tuple(joints[i]), tuple(joints[j]), (255, 255, 255), cfg.skeleton_width)

    # 画骨架点
    for point in joints:
        cv2.circle(image, tuple(point), cfg.skeleton_width, (255, 255, 255), -1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 显示图像
    # cv2.imshow('Skeleton', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return image

def batch_skeleton_map_gray(image_size, joints):
    images = []
    for joint in joints:
        image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

        # 将骨架点位转为整数坐标

        joints = np.array(joint, dtype=np.int32)
        # 定义骨架连接（根据你具体的骨架定义，这里是一个示例）
        if cfg.trainset == 'DEX_YCB':
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20)
            ]
        elif cfg.trainset == 'HO3D':
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 17),
                (0, 4), (4, 5), (5, 6), (6, 18),
                (0, 7), (7, 8), (8, 9), (9, 20),
                (0, 10), (10, 11), (11, 12), (12, 19),
                (0, 13), (13, 14), (14, 15), (15, 16)
            ]

        # 画骨架连接
        # for (i, j) in connections:
        #     cv2.line(image, tuple(joints[i]), tuple(joints[j]), (255, 255, 255), cfg.skeleton_width)

        # 画骨架点
        for point in joints:
            cv2.circle(image, tuple(point), cfg.skeleton_width, (255, 255, 255), -1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 显示图像
        # cv2.imshow('Skeleton', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        images.append(image)

    return torch.tensor(np.expand_dims(np.array(images), axis=1).astype(np.float32))/255.

if __name__=='__main__':
    skeletons = np.random.rand(21, 2)
    images = skeleton_map_gray((256, 256), skeletons)
    print(images)
    print(images.shape)

    skeletons = np.random.rand(2, 21, 2)
    images = batch_skeleton_map_gray((256, 256), skeletons)
    print(images)
    print(images.shape)
