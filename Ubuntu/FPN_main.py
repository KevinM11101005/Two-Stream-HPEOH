## FPN train + test process
import torch
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from main.config import cfg
from FPN_trainer import Trainer
from FPN_tester import Tester

cfg.set_args('0', False)
cudnn.benchmark = True

trainer = Trainer()
trainer._make_batch_generator()
trainer._make_model()

tester = Tester()
tester._make_batch_generator()

# train
for epoch in range(trainer.start_epoch, cfg.end_epoch):
    
    trainer.set_lr(epoch)
    trainer.tot_timer.tic()
    trainer.read_timer.tic()
    for itr, (inputs, targets) in enumerate(trainer.batch_generator):
        trainer.read_timer.toc()
        trainer.gpu_timer.tic()

        # forward
        trainer.optimizer.zero_grad()
        if cfg.backbone == 'crossatt' or cfg.simcc:
            loss, acc = trainer.model(inputs, targets, 'train')
        else:
            loss = trainer.model(inputs, targets, 'train')

        loss = {k:loss[k].mean() for k in loss}

        # backward
        sum(loss[k] for k in loss).backward()
        trainer.optimizer.step()
        trainer.gpu_timer.toc()
        screen = [
            'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
            'lr: %g' % (trainer.get_lr()),
            'speed: %.2f(gpu%.2fs r_data%.2fs)s/itr' % (
                trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
            '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
            ]
        screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k,v in loss.items()]
        if cfg.backbone == 'crossatt' or cfg.simcc:
            screen += ['%s: %.4f' % ('acc_' + k, v.detach()) for k,v in acc.items()]
        trainer.logger.info(' '.join(screen))

        trainer.tot_timer.toc()
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
    
    if (epoch+1)%cfg.ckpt_freq== 0 or epoch+1 == cfg.end_epoch:
        trainer.save_model({
            'epoch': epoch,
            'network': trainer.model.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
        }, epoch+1)

        tester._make_model(epoch+1)

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