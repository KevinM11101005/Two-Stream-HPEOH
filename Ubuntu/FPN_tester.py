## FPN Tester
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn.parallel.data_parallel import DataParallel
from Base_model import Base
from main.config import cfg
from DEX_YCB import DEX_YCB
from HO3D import HO3D
# dynamic model import
if cfg.backbone == 'fpn':
    from main.model import get_model
    
class Tester(Base):
    def __init__(self):
        super(Tester, self).__init__(log_name = 'test_logs.txt')

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        # Augment train data
        # test_transforms = transforms.Compose([
        #     transforms.Resize((256, 192)),
        #     transforms.ToTensor()
        # ])
        self.test_dataset = eval(cfg.testset)(transforms.ToTensor(), "test")
        self.batch_generator = DataLoader(dataset=self.test_dataset, batch_size=cfg.num_gpus*cfg.test_batch_size, shuffle=False, num_workers=cfg.num_thread, pin_memory=True)
       
    def _make_model(self, test_epoch):
        model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth.tar' % test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))
        
        # prepare network
        self.logger.info("Creating graph...")
        model = get_model('test')
        model = DataParallel(model).cuda()
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'], strict=False)
        model.eval()

        self.model = model

    def _evaluate(self, outs, cur_sample_idx):
        eval_result = self.test_dataset.evaluate(outs, cur_sample_idx)
        return eval_result

    def _print_eval_result(self, test_epoch):
        message = self.test_dataset.print_eval_result(test_epoch)
        for msg in message:
            self.logger.info(msg)
