import argparse
import torch
import logging
def get_logger():
    logger =  logging.getLogger()
    handler =  logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s%(levelname)s\t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def clip_gradient(optimizer,grad_clip):
    #梯度裁剪
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip,grad_clip)
def save_checkpoint(epoch,epoch_since_improvement,model,optimizer,acc,is_best):
    state={
        'epoch':epoch,
        'epoch_since_improvement':epoch_since_improvement,
        'model':model,
        'acc':acc,
        'optimizer':optimizer
    }
    save_filname='chechkpoin.tar'
    torch.save(state,save_filname)
    if is_best:
        torch.save(state,'BEST_checkpoint.tar')

class AverageMeter(object):
    def __init__(self):
        # self.val = 0
        # self.avg = 0
        # self.sum = 0
        # self.count = 0
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self,val,n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum /self.count


class LossMeterBag(object):
    def __init__(self,name_list):
        self.meter_dict = dict()
        self.name_list = name_list
        for name in self.name_list:
            self.meter_dict[name] = AverageMeter()
    
    def update(self,val_list):
        for i, name in enumerate(self.name_list):
            val = val_list[i]
            self.meter_dict[name].update(val)

    def __str__(self):
        ret = ''
        for name in self.name_list:
            ret += '{0}:\t{1:.4f}({2:.4f})\t'.format(name,self.meter_dict[name].val,self.meter_dict[name].avg)
        
        return ret

def adjust_learning_rate(optimizer,shrink_factor):

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    
    print('new lr is :%f'%(optimizer.param_groups[0]['lr']))

def get_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']

def accuracy(scores, target, k=1):
    batch_size = target.size(0)
    _, ind = scores.topk(k,1,True,True)
    correct = ind.eq(target.view(-1,1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()
    return correct_total.item() *(100./batch_size)

def parse_args():

     parser = argparse.ArgumentParser(description='train face network')

     parser.add_argument('--end_epoch', type=int, default=10, help='training epoch size')
     parser.add_argument('--lr', type=float, default=0.005, help='start learning rate')
     parser.add_argument('--weight_decay',type=float, default=0.0, help='weight decay')
     parser.add_argument('--batch_size',type=int, default=32,help='batch size in each context')
     parser.add_argument('--checkpoint',type=str,default=None,help='checkpoint')
     
     args = parser.parse_args()
     return args

