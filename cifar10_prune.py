import os
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm
import argparse

from pruner import check_sparsity, global_prune_model
from misc import set_seed, warmup_lr
from cfg import *

p = argparse.ArgumentParser()
p.add_argument('--network', choices=["resnet18"], default="resnet18")
p.add_argument('--seed', type=int, default=777)
p.add_argument('--num-workers', type=int, default=2)
p.add_argument('--score-type', type=str, choices=["grasp", "synflow", "synflow_iterative"], default=None)
p.add_argument('--prune-ratio', type=float, default=0.) 
args = p.parse_args()

set_seed(args.seed)

assert 0 <= args.prune_ratio < 1
batch_size = 256
data_path = os.path.join(data_path, "CIFAR10")
if args.prune_ratio == 0:
    save_path = os.path.join(results_path, args.network, "dense", f"seed{args.seed}")
else:
    assert args.score_time != None
    save_path = os.path.join(results_path, args.network, args.score_type, f"ratio{str(args.prune_ratio).replace('.', 'p')}", f"seed{args.seed}")
epochs = 182
warmup_epoch = 1
lr = 0.1
weight_decay = 5e-4
momentum = 0.9
decreasing_lr = [91, 136]

# Data
mean = [0.4914, 0.4822, 0.4465]
std = [0.2471, 0.2435, 0.2616]
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
train_data = CIFAR10(root = data_path, train = True, download = False, transform = train_transform)
train_loader = DataLoader(train_data, batch_size, shuffle = True, num_workers=args.num_workers)
test_data = CIFAR10(root = data_path, train = False, download = False, transform = test_transform)
test_loader = DataLoader(test_data, batch_size, shuffle = False, num_workers=args.num_workers)

# Network
network = eval(args.network)(num_classes = 10)
network.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
network.maxpool = nn.Identity()
network = network.cuda()

# Prune
if args.prune_ratio != 0:
    global_prune_model(network, args.prune_ratio, args.score_type, train_loader)
    check_sparsity(network, if_print=True)

# Optimizer
optimizer = optim.SGD(network.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

# Make Dir
os.makedirs(save_path, exist_ok=True)
logger = SummaryWriter(os.path.join(save_path, 'tensorboard'))

# Train
best_acc = 0. 
scaler = GradScaler()
for epoch in range(epochs):
    network.train()
    total_num = 0
    true_num = 0
    loss_sum = 0
    pbar = tqdm(train_loader, total=len(train_loader),
            desc=f"Train Epo {epoch} Lr {optimizer.param_groups[0]['lr']:.1e}", ncols=100)
    for i, (x, y) in enumerate(pbar):
        if epoch < warmup_epoch:
            warmup_lr(optimizer, epoch, i+1, warmup_epoch, len(train_loader), lr)
        x, y = x.cuda(), y.cuda()
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
    logger.add_scalar("train/acc", true_num/total_num, epoch)
    logger.add_scalar("train/loss", loss_sum/total_num, epoch)
    scheduler.step()

    # Test
    network.eval()
    fxs = []
    ys = []
    pbar = tqdm(test_loader, total=len(test_loader), desc=f"Evaluating", ncols=100)
    for x, y in pbar:
        x = x.cuda()
        with torch.no_grad(), autocast():
            fx = network(x)
        fxs.append(fx)
        ys.append(y)
    fxs = torch.cat(fxs).cpu().float()
    ys = torch.cat(ys).cpu()
    acc = torch.argmax(fxs, -1).eq(ys).float().mean()
    logger.add_scalar("test/acc", acc, epoch)

    # Save CKPT
    state_dict = {
        "network_dict": network.state_dict(),
        "optimizer_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
    }
    if acc > best_acc:
        best_acc = acc
        state_dict['best_acc'] = acc
        torch.save(state_dict, os.path.join(save_path, 'best.pth'))
    torch.save(state_dict, os.path.join(save_path, 'ckpt.pth'))