import numpy as np
import torch
import random

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True

def warmup_lr(optimizer, epoch, step, warmup_epoch, one_epoch_step, lr):

    overall_steps = warmup_epoch*one_epoch_step
    current_steps = epoch*one_epoch_step + step 

    new_lr = lr * current_steps/overall_steps
    new_lr = min(new_lr, lr)

    for p in optimizer.param_groups:
        p['lr']=new_lr