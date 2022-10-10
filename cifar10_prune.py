import os
import torch
import numpy as np
from torch import nn
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
from process import train_once, test_once
from misc import set_seed
from cfg import *

p = argparse.ArgumentParser()
p.add_argument('--network', choices=["resnet20"], default="resnet20")
p.add_argument('--seed', type=int, default=7)
p.add_argument('--num-workers', type=int, default=2)
p.add_argument('--score-type', type=str, choices=["mp", "grasp", "synflow", "synflow_iterative"], default="mp")
p.add_argument('--prune-ratio', type=float, default=0.) 
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(args.seed)

assert 0 <= args.prune_ratio < 1
batch_size = 256
data_path = os.path.join(data_path, "CIFAR10")
if args.prune_ratio == 0:
    save_path = os.path.join(results_path, args.network, "dense", f"seed{args.seed}")
else:
    assert args.score_type != None
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
if args.network in ["resnet18"]:
    network = eval(args.network)(num_classes = 10)
    network.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    network.maxpool = nn.Identity()
    network = network.to(device)
elif args.network in ["resnet20"]:
    network = eval(args.network)()
    network = network.to(device)

# Prune
if args.prune_ratio != 0:
    global_prune_model(network, args.prune_ratio, args.score_type, train_loader)
    check_sparsity(network, if_print=True)

# Optimizer
optimizer = torch.optim.SGD(network.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

# Make Dir
os.makedirs(save_path, exist_ok=True)
logger = SummaryWriter(os.path.join(save_path, 'tensorboard'))

# Train
best_acc = 0. 
scaler = GradScaler()
for epoch in range(epochs):
    train_acc, train_loss = train_once(network=network, train_loader=train_loader, optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch, warmup_epoch=warmup_epoch, lr=lr)
    logger.add_scalar("train/acc", train_acc, epoch)
    logger.add_scalar("train/loss", train_loss, epoch)
    test_acc = test_once(network=network, test_loader=test_loader, epoch=epoch)
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
    torch.save(state_dict, os.path.join(save_path, 'ckpt.pth'))