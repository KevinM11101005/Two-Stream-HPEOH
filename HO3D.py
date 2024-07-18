import os
import os.path as osp
import numpy as np
import torch
import cv2
import random
import json
import time
import torchvision.transforms as transforms
import copy
from common.logger import colorlogger
from common.utils.skeleton_map import skeleton_map_gray
from pycocotools.coco import COCO
from main.config import cfg
from common.utils.preprocessing import load_img, get_bbox, process_bbox, generate_patch_image, augmentation
from common.utils.transforms import cam2pixel, compute_mpjpe, compute_pa_mpjpe
from common.codecs.keypoint_eval import keypoint_pck_accuracy
from common.utils.vis import ux_hon_result, ux_hon_result_final

# 設置隨機種子
random.seed(cfg.random_seed)

class HO3D(torch.utils.data.Dataset):
    def __init__(self, transform, data_split, log_name='cfg_logs.txt'):
        self.transform = transform
        self.data_split = data_split if data_split == 'train' else 'evaluation'
        self.root_dir = osp.join('data', 'HO3D')
        self.annot_path = osp.join(self.root_dir, 'annotations')
        self.root_joint_idx = 0
        self.datalist = self.load_data()
        if self.data_split != 'train':
            self.eval_result = [[],[],[],[]] #[mpjpe_list, pa-mpjpe_list]
        
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)
        
        for i in cfg.__dict__:
            self.logger.info('{0}: {1}'.format(i, cfg.__dict__[i]))
        message = []
        message.append(f"DataList len: {len(self.datalist)}")
        message.append('left hand data: {0}, right hand data: {1}'.format(0, len(self.datalist)))
        
        if cfg.simcc and cfg.SET:
            message.append(f'Start the model {cfg.backbone} with SET and simcc')
        elif cfg.simcc:
            message.append(f'Start the model {cfg.backbone} without SET and with simcc')
        elif cfg.SET:
            message.append(f'Start the model {cfg.backbone} with SET and with regressor')
        else:
            message.append(f'Start the model {cfg.backbone} without SET and simcc')
        for msg in message:
            self.logger.info(msg)
            
    def load_data(self):
        db = COCO(osp.join(self.annot_path, "HO3D_train_data.json"))
        # db = COCO(osp.join(self.annot_path, 'HO3Dv3_partial_test_multiseq_coco.json'))

        datalist = []
        skip = 1
        skip_mode = cfg.train_skip
        remainder = cfg.train_remainder

        for aid in db.anns.keys():
            if skip % skip_mode == remainder:
                ann = db.anns[aid]
                image_id = ann['image_id']
                img = db.loadImgs(image_id)[0]
                if osp.exists(osp.join(self.root_dir, 'train', img['file_name'])):
                    img_path = osp.join(self.root_dir, 'train', img['file_name'])
                    # TEMP
                    # img_path = osp.join(self.root_dir, 'train', img['sequence_name'], 'rgb', img['file_name'])

                    img_shape = (img['height'], img['width'])
                    joints_coord_cam = np.array(ann['joints_coord_cam'], dtype=np.float32) # meter
                    cam_param = {k:np.array(v, dtype=np.float32) for k,v in ann['cam_param'].items()}
                    joints_coord_img = cam2pixel(joints_coord_cam, cam_param['focal'], cam_param['princpt'])
                    bbox = get_bbox(joints_coord_img[:,:2], np.ones_like(joints_coord_img[:,0]), expansion_factor=1.5)
                    bbox = process_bbox(bbox, img['width'], img['height'], expansion_factor=1.0)
                    data = {"img_path": img_path, "img_shape": img_shape, "joints_coord_img": joints_coord_img,
                            "bbox": bbox,}
                        
                    if all(val is not None for val in data.values()):
                        datalist.append(data)
            skip += 1

        # 確定抽樣大小
        n = len(datalist)
        print(f'Total DataList len: {n}')
        train_size = int(cfg.train_size * n)
        test_size = n - train_size

        # 隨機抽樣
        train_samples = random.sample(datalist, train_size)
        test_samples = [item for item in datalist if item not in train_samples]
        if self.data_split == 'train':
            return train_samples
        else:
            return test_samples
        
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox = data['img_path'], data['img_shape'], data['bbox']

        # img
        img = load_img(img_path)
        img, img2bb_trans, self.bb2img_trans, rot, scale = augmentation(img, bbox, self.data_split, do_flip=False)
        img = self.transform(img.astype(np.float32))/255.

        if self.data_split == 'train':
            targets = {}
            ## 2D joint coordinate
            joints_img = data['joints_coord_img']
            joints_img_xy1 = np.concatenate((joints_img[:,:2], np.ones_like(joints_img[:,:1])),1)
            joints_img = np.dot(img2bb_trans, joints_img_xy1.transpose(1,0)).transpose(1,0)[:,:2]
            if not cfg.simcc:
                joints_img_copy = joints_img.copy()
                ## normalize to [0,1]
                joints_img_copy[:,0] /= cfg.input_img_shape[0]
                joints_img_copy[:,1] /= cfg.input_img_shape[1]
                targets['joints_img'] = joints_img_copy
            else:
                targets['joints_img'] = joints_img
            
            skeleton_map = skeleton_map_gray((cfg.input_img_shape[0], cfg.input_img_shape[1]), joints_img)
            skeleton_map = self.transform(skeleton_map.astype(np.float32))/255.

            inputs = {'img': img}
            targets['skeleton_map'] = skeleton_map
        else:
            inputs = {'img': img}
            targets = {}

        return inputs, targets
                  
    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        for n in range(sample_num):            
            annot = annots[cur_sample_idx + n]
            # cv2.namedWindow(annot['img_path'], 0)
            
            out = outs[n]
            
            start_time = time.time()
            # img convert
            img = load_img(annot['img_path'])
            orig_img = copy.deepcopy(img)
            img, img2bb_trans, bb2img_trans, rot, scale = augmentation(img, annot['bbox'], self.data_split, do_flip=False)
        
    #         # GT and out['keypoints]
            gt_joints_coord_img = annot['joints_coord_img']
            joints_img_xy1 = np.concatenate((gt_joints_coord_img[:,:2], np.ones_like(gt_joints_coord_img[:,:1])),1)
            joints_img = np.dot(img2bb_trans, joints_img_xy1.transpose(1,0)).transpose(1,0)[:,:2]
            
            if cfg.backbone == 'unext':
                gt_skeleton_map = skeleton_map_gray((cfg.input_img_shape[0], cfg.input_img_shape[1]), joints_img)
                gt_skeleton_map = gt_skeleton_map/255.
                
                pred_skeleton_map = (out['skeleton_map'].squeeze().cpu().numpy()).astype(float)# > 0.5
                
                ## show result
                cat_imgs = ux_hon_result(orig_img, img, pred_skeleton_map, gt_skeleton_map)
                cat_imgs = ux_hon_result_final(out, bb2img_trans, orig_img, img, cat_imgs)
                
                path = cfg.vis_dir+'/'+'_'.join(annot['img_path'].split('/'))
                cv2.imwrite(path, cat_imgs)
                # cv2.imshow(annot['img_path'], cat_imgs)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                pred_skeleton_map = (out['skeleton_map'].squeeze().cpu().numpy()>0.5).astype(float)# > 0.5
 
                num_correct = (pred_skeleton_map==gt_skeleton_map).sum()
                num_pixels = cfg.input_img_shape[0] * cfg.input_img_shape[1]
            else:
                img_uint8 = cv2.resize(orig_img.astype(np.uint8), (cfg.input_img_shape[0], cfg.input_img_shape[1]))
                rgb_img_uint8 = cv2.cvtColor(img_uint8.astype(np.uint8), cv2.COLOR_BGR2RGB)
                rgb_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
                ori_imgs = np.hstack([rgb_img_uint8, rgb_img])
                cat_imgs = ux_hon_result_final(out, bb2img_trans, orig_img, img, ori_imgs)
                
                path = cfg.vis_dir+'/'+'_'.join(annot['img_path'].split('/'))
                cv2.imwrite(path, cat_imgs)
                # cv2.imshow(annot['img_path'], cat_imgs)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

            if cfg.simcc:
                pred_keypoints = np.expand_dims(out['keypoints'], axis=0)
                pred_keypoint_scores = out['keypoint_scores']
                keypoints_restored = np.dot(bb2img_trans, np.concatenate((pred_keypoints[0], np.ones((pred_keypoints[0].shape[0], 1))), axis=1).transpose(1, 0))
                keypoints_restored = keypoints_restored[:2, :].transpose(1, 0)
            else:
                pred_keypoints = np.expand_dims(out['joints_coord_img'].cpu().numpy(), axis=0)
                pred_keypoints[:,:,0] *= cfg.input_img_shape[1]
                pred_keypoints[:,:,1] *= cfg.input_img_shape[0]
                keypoints_restored = np.dot(bb2img_trans, np.concatenate((pred_keypoints[0], np.ones((pred_keypoints[0].shape[0], 1))), axis=1).transpose(1, 0))
                keypoints_restored = keypoints_restored[:2, :].transpose(1, 0)
                pred_keypoint_scores = np.any(keypoints_restored, axis=1)

            end_time = time.time()
    #         # flip back to left hand
    #         if annot['hand_type'] == 'left':
    #             joints_out[:,0] *= -1 
            _, avg_acc, _ = keypoint_pck_accuracy(
                pred=np.expand_dims(keypoints_restored, axis=0),
                gt=np.expand_dims(gt_joints_coord_img[:,:2], axis=0),
                mask=np.expand_dims(pred_keypoint_scores, axis=0) > 0,
                thr=cfg.pck_thr,
                norm_factor=np.expand_dims(annot['img_shape'], axis=0),
            )

            self.eval_result[2].append(compute_mpjpe(keypoints_restored, gt_joints_coord_img[:,:2]))
            self.eval_result[3].append(end_time-start_time)

            if cfg.backbone == 'unext':
                self.eval_result[0].append(num_correct / num_pixels)
                self.eval_result[1].append(avg_acc)
            else:
                self.eval_result[0].append(avg_acc)
                
    def print_eval_result(self, test_epoch):
        message = []
        if cfg.backbone == 'unext':
            message.append('Output: {0}, Model: snapshot_{1}.pth.tar'.format(cfg.output_dir.split('\\')[-1], test_epoch))
            message.append('Correct/Total(One Batch) pixels: {0:.2f}'.format(np.mean(self.eval_result[0]) * 100))
            message.append('PCK@{0}: {1:.2f}'.format(cfg.pck_thr, np.mean(self.eval_result[1]) * 100))
            message.append('MPJPE : %.2f' % np.mean(self.eval_result[2]))
            message.append('Per image cost time : %.4f image/s' % np.mean(self.eval_result[3]))
        else:
            message.append('Output: {0}, Model: snapshot_{1}.pth.tar'.format(cfg.output_dir.split('\\')[-1], test_epoch))
            message.append('PCK@{0}: {1:.2f}'.format(cfg.pck_thr, np.mean(self.eval_result[0]) * 100))
            message.append('MPJPE : %.2f' % np.mean(self.eval_result[2]))
            message.append('Per image cost time : %.4f image/s' % np.mean(self.eval_result[3]))
        return message
    
if __name__=='__main__':
    train_data = HO3D(transforms.ToTensor(), "train")
    test_data = HO3D(transforms.ToTensor(), "test")
    print(train_data[0])
    print(test_data[0])