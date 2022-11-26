import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import argparse

from models.resnet_s import resnet20
from pruner import check_sparsity, global_prune_model
from process import train
from misc import set_seed
from cfg import *

p = argparse.ArgumentParser()
p.add_argument('--network', choices=["resnet18", "resnet20"], default="resnet18")
p.add_argument('--seed', type=int, default=7)
p.add_argument('--num-workers', type=int, default=2)
p.add_argument('--score-type', type=str, choices=["snip", "grasp", "synflow"], default="synflow")
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
    save_path = os.path.join(results_path, args.network, args.score_type, f"ratio{args.prune_ratio}", f"seed{args.seed}")
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

# Make Dir
os.makedirs(save_path, exist_ok=True)
logger = SummaryWriter(os.path.join(save_path, 'tensorboard'))

# Prune
if args.prune_ratio != 0:
    global_prune_model(network, args.prune_ratio, args.score_type, train_loader)
    check_sparsity(network, if_print=True)

# Train
train(network=network, train_loader=train_loader, test_loader=test_loader, 
    logger=logger, save_path=save_path, epochs=epochs, lr=lr, weight_decay=weight_decay, 
    omentum=momentum, decreasing_lr=decreasing_lr, warmup_epoch=warmup_epoch)