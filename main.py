import os
import torch
from torch import nn
from torchvision.models import resnet18
from torch.utils.tensorboard import SummaryWriter
import argparse

from models.resnet_s import resnet20
from dataset import prepare_dataset
from pruner import check_sparsity, global_prune_model
from process import train
from misc import set_seed, gen_folder_name
from cfg import *

p = argparse.ArgumentParser()
p.add_argument('--network', choices=["resnet18", "resnet20"], default="resnet20")
p.add_argument('--seed', type=int, default=7)
p.add_argument('--num-workers', type=int, default=2)
p.add_argument('--score-type', type=str, choices=["snip", "grasp", "synflow"], required=True)
p.add_argument('--dataset', type=str, choices=["cifar10", "cifar100", "flowers102", "oxfordpets", "country211"], required=True)
p.add_argument('--prune-ratio', type=float, default=0.) 
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(args.seed)

assert 0 <= args.prune_ratio < 1
batch_size = 512
save_path = os.path.join(results_path, gen_folder_name(args))

epochs = 182
lr = 0.1
weight_decay = 5e-4
momentum = 0.9
decreasing_lr = [91, 136]

train_loader, test_loader, cls_num = prepare_dataset(args.dataset, data_path)

# Network
if args.network in ["resnet18"]:
    network = eval(args.network)(num_classes = cls_num)
    if args.dataset in ["cifar10", "cifar100"]:
        network.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        network.maxpool = nn.Identity()
    network = network.to(device)
elif args.network in ["resnet20"]:
    network = eval(args.network)(cls_num)
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
    momentum=momentum, decreasing_lr=decreasing_lr)