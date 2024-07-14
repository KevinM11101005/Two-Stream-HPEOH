import os
import os.path as osp
import sys
import numpy as np

class Config:
    def __init__(self):
        ## dataset
        # HO3D, DEX_YCB
        self.trainset = 'DEX_YCB' #'HO3D'
        self.testset = 'DEX_YCB'
        self.train_skip = 1000
        self.test_skip = 1000
        self.train_remainder = 0
        self.test_remainder = 0
        
        ## input, output
        self.input_img_shape = (256,256)

        ## 'fpn', 'unext'
        self.backbone = 'fpn'
        self.skeleton_width = 5
        
        ## model
        self.SET = True
        self.simcc = False

        ## fpn att => 'SG', 'cbam'
        self.att = 'SG'

        ## cbam
        self.mask = False
        
        ## training config
        if self.trainset == 'DEX_YCB':
            self.lr_dec_epoch = [10*i for i in range(1,7)]
            self.end_epoch = 50
            if self.backbone == 'unext':
                self.ux_lr = 1e-4
                self.hon_lr = 1e-4
                self.skeleton_width = 5
            else:
                self.lr = 1e-4
            self.lr_dec_factor = 0.9
            ## save model
            self.ckpt_freq = 10
        elif self.trainset == 'HO3D':
            self.lr_dec_epoch = [10*i for i in range(1,7)]
            self.end_epoch = 50
            if self.backbone == 'unext':
                self.ux_lr = 1e-4
                self.hon_lr = 1e-4
                self.skeleton_width = 5
            else:
                self.lr = 1e-4
            self.lr_dec_factor = 0.7
            ## save model
            self.ckpt_freq = 10

        self.train_batch_size = 16 # per GPU
        
        ## final_layer
        self.in_channels = 256
        self.out_channels = 21
        self.final_layer_kernel_size = 3
        ## mlp
        self.flatten_dims = 1024
        ## gau
        self.gau_cfg = dict(
                hidden_dims=256,
                s=128,
                expansion_factor=2,
                dropout_rate=0.,
                drop_path=0.,
                act_fn='ReLU',
                use_rel_bias=False,
                pos_enc=False)
        ## codec
        self.codec = dict(
                type='SimCCLabel',
                input_size=(256, 256),
                sigma=(5.66, 5.66),
                simcc_split_ratio=2.0,
                normalize=False,
                use_dark=False)
        ## cls_x, cls_y
        self.W = int(self.codec['input_size'][0] * self.codec['simcc_split_ratio'])
        self.H = int(self.codec['input_size'][1] * self.codec['simcc_split_ratio'])

        ## SimCC loss
        self.loss=dict(
                type='KLDiscretLoss',
                use_target_weight=True,
                beta=10.,
                label_softmax=True)
        
        ## Heatmap loss
        self.lambda_joints_img = 100
        
        ## testing config
        self.test_batch_size = 64
        self.pck_thr = 0.05
        
        ## others
        self.num_thread = 0
        self.gpu_ids = '0'
        self.num_gpus = 1
        self.continue_train = False
        
        if self.SET:
            self.outputname = 'output_' + self.backbone + '_' + self.att + '_' + 'wSET'
        else:
            self.outputname = 'output_' + self.backbone + '_' + self.att + '_' + 'woSET'
            
        if self.backbone == 'unext' and self.SET:
            self.outputname = 'output_UX_HON' + '_' + self.att + '_' + 'wSET'
        elif self.backbone == 'unext' and not self.SET:
            self.outputname = 'output_UX_HON' + '_' + self.att + '_' + 'woSET'
            
        if self.simcc:
            self.outputname = self.outputname + '_simcc' + '_' + str(self.train_remainder)
        else:
            self.outputname = self.outputname + '_reg' + '_' + str(self.train_remainder)
              
        ## directory
        self.cur_dir = osp.dirname(os.path.abspath(__file__))
        self.root_dir = osp.join(self.cur_dir, '..')
        self.data_dir = osp.join(self.root_dir, 'data')
        self.output_dir = osp.join(self.root_dir, self.outputname)
        self.model_dir = osp.join(self.output_dir, 'model_dump')
        self.vis_dir = osp.join(self.output_dir, 'vis')
        self.log_dir = osp.join(self.output_dir, 'log')
        self.result_dir = osp.join(self.output_dir, 'result')
        self.mano_path = osp.join(self.root_dir, 'common', 'utils', 'manopth')
        
    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        # self.num_gpus = len(self.gpu_ids.split(','))
        self.num_gpus = len(self.gpu_ids)
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))

cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from common.utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
add_pypath(osp.join(cfg.data_dir, cfg.trainset))
add_pypath(osp.join(cfg.data_dir, cfg.testset))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)
