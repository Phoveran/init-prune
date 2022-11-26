from tqdm import tqdm
import torch
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
import os

def train(network, train_loader, test_loader, logger, save_path, epochs, lr, weight_decay, momentum, decreasing_lr):
    # Optimizer
    optimizer = torch.optim.SGD(network.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)
    best_acc = 0. 
    scaler = GradScaler()
    for epoch in range(epochs):
        pbar = tqdm(train_loader, total=len(train_loader),
                desc=f"Train Epo {epoch} Lr {optimizer.param_groups[0]['lr']:.1e}", ncols=100)
        train_acc, train_loss = train_once(network=network, pbar=pbar, optimizer=optimizer, scheduler=scheduler, scaler=scaler)
        logger.add_scalar("train/acc", train_acc, epoch)
        logger.add_scalar("train/loss", train_loss, epoch)
        test_acc = test_once(network=network, test_loader=test_loader)
        logger.add_scalar("test/acc", test_acc, epoch)
        # Save CKPT
        state_dict = {
            "network_dict": network.state_dict(),
            "optimizer_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
        }
        if test_acc > best_acc:
            best_acc = test_acc
            state_dict['best_acc'] = test_acc
            torch.save(state_dict, os.path.join(save_path, 'best.pth'))
            torch.save(network.state_dict(), os.path.join(save_path, 'state_dict.pth'))
        torch.save(state_dict, os.path.join(save_path, 'ckpt.pth'))

def train_once(network, pbar, optimizer, scheduler, scaler):
    device = next(network.parameters()).device
    network.train()
    total_num = 0
    true_num = 0
    loss_sum = 0
    for x, y in pbar:
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

def test_once(network, test_loader):
    device = next(network.parameters()).device
    network.eval()
    fxs = []
    ys = []
    pbar = tqdm(test_loader, total=len(test_loader), desc=f"Evaluating", ncols=100)
    for x, y in pbar:
        x = x.to(device)
        with torch.no_grad():
            fx = network(x)
        fxs.append(fx)
        ys.append(y)
    fxs = torch.cat(fxs).cpu().float()
    ys = torch.cat(ys).cpu()
    acc = torch.argmax(fxs, -1).eq(ys).float().mean()
    pbar.set_postfix_str(f"Acc {100*acc:.2f}%")
    return acc