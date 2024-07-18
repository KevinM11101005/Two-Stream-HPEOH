import abc
from common.timer import Timer
from common.logger import colorlogger
from main.config import cfg
# dynamic model import
if cfg.backbone == 'fpn':
    from main.model import get_model
elif cfg.backbone == 'conv': 
    from main.model_convnext import get_model
elif cfg.backbone == 'crossfit': 
    from main.model_crossFIT import get_model
elif cfg.backbone == 'crossatt': 
    from main.model_crossatt import get_model
elif cfg.backbone == 'cross_res_hr': 
    from main.model_cross_res_hrnet import get_model
elif cfg.backbone == 'unext': 
    from main.model_UNext import get_model

# dynamic dataset import
# exec('from ' + cfg.trainset + ' import ' + cfg.trainset)
# exec('from ' + cfg.testset + ' import ' + cfg.testset)

class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name='logs.txt'):
        
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return