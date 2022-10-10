from tqdm import tqdm
from misc import warmup_lr
import torch
from torch.cuda.amp import autocast
from torch.nn import functional as F

def train_once(network, train_loader, optimizer, scheduler, scaler, epoch, warmup_epoch, lr):
    device = next(network.parameters()).device
    network.train()
    total_num = 0
    true_num = 0
    loss_sum = 0
    pbar = tqdm(train_loader, total=len(train_loader),
            desc=f"Train Epo {epoch} Lr {optimizer.param_groups[0]['lr']:.1e}", ncols=100)
    for i, (x, y) in enumerate(pbar):
        if epoch < warmup_epoch:
            warmup_lr(optimizer, epoch, i+1, warmup_epoch, len(train_loader), lr)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with autocast():
            fx = network(x)
            loss = F.cross_entropy(fx, y, reduction='mean')
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_num += y.size(0)
        true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
        loss_sum += loss.item() * fx.size(0)
        pbar.set_postfix_str(f"Acc {100*true_num/total_num:.2f}%")
    scheduler.step()
    return true_num/total_num, loss/total_num

def test_once(network, test_loader, epoch):
    device = next(network.parameters()).device
    network.eval()
    fxs = []
    ys = []
    pbar = tqdm(test_loader, total=len(test_loader), desc=f"Evaluating", ncols=100)
    for x, y in pbar:
        x = x.to(device)
        with torch.no_grad(), autocast():
            fx = network(x)
        fxs.append(fx)
        ys.append(y)
    fxs = torch.cat(fxs).cpu().float()
    ys = torch.cat(ys).cpu()
    acc = torch.argmax(fxs, -1).eq(ys).float().mean()
    return acc