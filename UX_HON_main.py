## Unext + HandOccNet train + test process
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from main.config import cfg
from UX_HON_trainer import UH_Trainer
from UX_HON_tester import UH_Tester

cfg.set_args('0', False)
cudnn.benchmark = True

trainer = UH_Trainer()
trainer._make_batch_generator()
trainer._make_UX_model()
trainer._make_HON_model()

tester = UH_Tester()
tester._make_batch_generator()

# train
for epoch in range(trainer.start_epoch, cfg.end_epoch):
    
    trainer.set_ux_lr(epoch)
    trainer.set_hon_lr(epoch)
    trainer.tot_timer.tic()
    trainer.read_timer.tic()
    for itr, (inputs, targets) in enumerate(trainer.batch_generator):
        trainer.read_timer.toc()
        trainer.gpu_timer.tic()

        ux_loss, ux_acc, ux_outs = trainer.ux_model(inputs, targets, 'train', itr)

        ux_loss_dic = {k:ux_loss[k].mean() for k in ux_loss}
        ux_loss = sum(ux_loss[k] for k in ux_loss_dic)

        # ux forward
        trainer.ux_optimizer.zero_grad()
        # ux backward
        ux_loss.backward(retain_graph=True)
        trainer.ux_optimizer.step()

        if cfg.simcc:
            hon_loss, hon_acc = trainer.hon_model(inputs, targets, ux_outs, 'train')
        else:
            hon_loss = trainer.hon_model(inputs, targets, 'train')

        hon_loss_dic = {k:hon_loss[k].mean() for k in hon_loss}
        hon_loss = sum(hon_loss[k] for k in hon_loss_dic)

        # hon forward
        trainer.hon_optimizer.zero_grad()
        # hon backward
        hon_loss.backward()
        trainer.hon_optimizer.step()

        trainer.gpu_timer.toc()
        screen = [
            'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
            'ux_lr: %g' % (trainer.get_ux_lr()),
            'hon_lr: %g' % (trainer.get_hon_lr()),
            'speed: %.2f(gpu%.2fs r_data%.2fs)s/itr' % (
                trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
            '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
            ]
        screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k,v in ux_loss_dic.items()]
        screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k,v in hon_loss_dic.items()]
        
        screen += ['%s: %.4f' % ('acc_' + k, v.detach()) for k,v in ux_acc.items()]
        if cfg.simcc:
            screen += ['%s: %.4f' % ('acc_' + k, v.detach()) for k,v in hon_acc.items()]
        trainer.logger.info(' '.join(screen))

        trainer.tot_timer.toc()
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
    
    if (epoch+1)%cfg.ckpt_freq== 0 or epoch+1 == cfg.end_epoch:
        trainer.save_UX_model({
            'epoch': epoch,
            'network': trainer.ux_model.state_dict(),
            'optimizer': trainer.ux_optimizer.state_dict(),
        }, epoch+1)

        trainer.save_HON_model({
            'epoch': epoch,
            'network': trainer.hon_model.state_dict(),
            'optimizer': trainer.hon_optimizer.state_dict(),
        }, epoch+1)

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