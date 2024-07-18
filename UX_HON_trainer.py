## UneXt + HandOccNet UH_Trainer
import os.path as osp
import glob
import math
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn.parallel.data_parallel import DataParallel
from Base_model import Base
from main.config import cfg
from DEX_YCB import DEX_YCB
from HO3D import HO3D
from main.model_UX import get_UX_model
from main.model_HON import get_HON_model

class UH_Trainer(Base):
    def __init__(self):
        super(UH_Trainer, self).__init__(log_name = 'train_logs.txt')

    def get_UX_optimizer(self, model):
        # model_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.ux_lr)
        return optimizer

    def get_HON_optimizer(self, model):
        # model_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.hon_lr)
        return optimizer

    def save_UX_model(self, state, epoch):
        file_path = osp.join(cfg.model_dir,'snapshot_UX_{}.pth.tar'.format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info("Write UX snapshot into {}".format(file_path))

    def save_HON_model(self, state, epoch):
        file_path = osp.join(cfg.model_dir,'snapshot_HON_{}.pth.tar'.format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info("Write HON snapshot into {}".format(file_path))

    def load_UX_model(self, model, optimizer):
        model_file_list = glob.glob(osp.join(cfg.model_dir,'*.pth.tar'))
        cur_epoch = max([int(file_name[file_name.find('snapshot_UX_') + 9 : file_name.find('.pth.tar')]) for file_name in model_file_list])
        ckpt_path = osp.join(cfg.model_dir, 'snapshot_UX_' + str(cur_epoch) + '.pth.tar')
        ckpt = torch.load(ckpt_path) 
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['network'], strict=False)
        #optimizer.load_state_dict(ckpt['optimizer'])

        self.logger.info('Load checkpoint from {}'.format(ckpt_path))
        return start_epoch, model, optimizer

    def load_HON_model(self, model, optimizer):
        model_file_list = glob.glob(osp.join(cfg.model_dir,'*.pth.tar'))
        cur_epoch = max([int(file_name[file_name.find('snapshot_HON_') + 9 : file_name.find('.pth.tar')]) for file_name in model_file_list])
        ckpt_path = osp.join(cfg.model_dir, 'snapshot_HON_' + str(cur_epoch) + '.pth.tar')
        ckpt = torch.load(ckpt_path) 
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['network'], strict=False)
        #optimizer.load_state_dict(ckpt['optimizer'])

        self.logger.info('Load checkpoint from {}'.format(ckpt_path))
        return start_epoch, model, optimizer
    
    def set_ux_lr(self, epoch):
        for e in cfg.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < cfg.lr_dec_epoch[-1]:
            idx = cfg.lr_dec_epoch.index(e)
            for g in self.ux_optimizer.param_groups:
                g['lr'] = cfg.ux_lr * (cfg.lr_dec_factor ** idx)
        else:
            for g in self.ux_optimizer.param_groups:
                g['lr'] = cfg.ux_lr * (cfg.lr_dec_factor ** len(cfg.lr_dec_epoch))

    def set_hon_lr(self, epoch):
        for e in cfg.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < cfg.lr_dec_epoch[-1]:
            idx = cfg.lr_dec_epoch.index(e)
            for g in self.hon_optimizer.param_groups:
                g['lr'] = cfg.hon_lr * (cfg.lr_dec_factor ** idx)
        else:
            for g in self.hon_optimizer.param_groups:
                g['lr'] = cfg.hon_lr * (cfg.lr_dec_factor ** len(cfg.lr_dec_epoch))

    def get_ux_lr(self):
        for g in self.ux_optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr
    
    def get_hon_lr(self):
        for g in self.hon_optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr
    
    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        # Augment train data
        # train_transforms = transforms.Compose([
        #     transforms.Resize((256, 192)),
        #     transforms.ToTensor()
        # ])
        train_dataset = eval(cfg.trainset)(transforms.ToTensor(), "train")
            
        self.itr_per_epoch = math.ceil(len(train_dataset) / cfg.num_gpus / cfg.train_batch_size)
        self.batch_generator = DataLoader(dataset=train_dataset, batch_size=cfg.num_gpus*cfg.train_batch_size, shuffle=True, num_workers=cfg.num_thread, pin_memory=True)

    def _make_UX_model(self):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = get_UX_model('train')

        model = DataParallel(model).cuda()
        ux_optimizer = self.get_UX_optimizer(model)
        if cfg.continue_train:
            start_epoch, model, ux_optimizer = self.load_model(model, ux_optimizer)
        else:
            start_epoch = 0
        model.train()

        self.start_epoch = start_epoch
        self.ux_model = model
        self.ux_optimizer = ux_optimizer

    def _make_HON_model(self):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        if cfg.SET:
            self.logger.info("Creating model with SET...")
        else:
            self.logger.info("Creating model without SET...")
        model = get_HON_model('train')

        model = DataParallel(model).cuda()
        hon_optimizer = self.get_HON_optimizer(model)
        if cfg.continue_train:
            start_epoch, model, hon_optimizer = self.load_model(model, hon_optimizer)
        else:
            start_epoch = 0
        model.train()

        self.start_epoch = start_epoch
        self.hon_model = model
        self.hon_optimizer = hon_optimizer