## FPN test process
import torch
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from main.config import cfg
from FPN_tester import Tester

test_epoch = [epoch for epoch in range(cfg.ckpt_freq, cfg.end_epoch+1, cfg.ckpt_freq)]
cfg.set_args('0', False)
cudnn.benchmark = True

tester = Tester()
tester._make_batch_generator()

for epoch in test_epoch:
    tester._make_model(epoch)

    eval_result = {}
    cur_sample_idx = 0
    for itr, (inputs, targets) in enumerate(tqdm(tester.batch_generator)):
        
        # forward
        with torch.no_grad():
            out = tester.model(inputs, targets, 'test')
        
        # save output
        out = {k: v for k,v in out.items()}
        for k,v in out.items(): batch_size = out[k].shape[0]
        out = [{k: v[bid] for k,v in out.items()} for bid in range(batch_size)]

        # evaluate
        tester._evaluate(out, cur_sample_idx)
        cur_sample_idx += len(out)

    tester._print_eval_result(epoch)