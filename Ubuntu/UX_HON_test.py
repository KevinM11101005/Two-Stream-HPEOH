## Unext + HandOccNet test process
import torch
from tqdm import tqdm
from UX_HON_tester import UH_Tester
import torch.backends.cudnn as cudnn
from main.config import cfg

cfg.set_args('0', False)
cudnn.benchmark = True

tester = UH_Tester()
tester._make_batch_generator()

# train
for epoch in range(cfg.ckpt_freq-1, cfg.end_epoch, cfg.ckpt_freq):
    tester._make_UX_model(epoch+1)
    tester._make_HON_model(epoch+1)

    eval_result = {}
    cur_sample_idx = 0
    for itr, (inputs, targets) in enumerate(tqdm(tester.batch_generator)):
        
        # forward
        with torch.no_grad():
            ux_out = tester.ux_model(inputs, targets, 'test', itr)
            hon_out = tester.hon_model(inputs, targets, ux_out, 'test')
        
        # save output
        ux_out = {k: v for k,v in ux_out.items()}
        for k,v in ux_out.items(): batch_size = ux_out[k].shape[0]
        hon_out = {k: v for k,v in hon_out.items()}
        for k,v in hon_out.items(): batch_size = hon_out[k].shape[0]
        out = []
        for bid in range(batch_size):
            combined_dict = {}
            for k,v in ux_out.items():
                combined_dict[k] = v[bid]
            for k,v in hon_out.items():
                combined_dict[k] = v[bid]
            out.append(combined_dict)
            
        # evaluate
        tester._evaluate(out, cur_sample_idx)
        cur_sample_idx += len(out)

    tester._print_eval_result(epoch)