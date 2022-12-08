import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def prepare_dataset(dataset, data_path):
    data_path = os.path.join(data_path, dataset)
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2471, 0.2435, 0.2616]
    if dataset == "cifar10":
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
        train_data = datasets.CIFAR10(root = data_path, train = True, download = False, transform = train_transform)
        train_loader = DataLoader(train_data, 512, shuffle = True, num_workers=2)
        test_data = datasets.CIFAR10(root = data_path, train = False, download = False, transform = test_transform)
        test_loader = DataLoader(test_data, 512, shuffle = False, num_workers=2)
        cls_num = 10
    elif dataset == "cifar100":
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
        train_data = datasets.CIFAR100(root = data_path, train = True, download = True, transform = train_transform)
        train_loader = DataLoader(train_data, 512, shuffle = True, num_workers=2)
        test_data = datasets.CIFAR100(root = data_path, train = False, download = True, transform = test_transform)
        test_loader = DataLoader(test_data, 512, shuffle = False, num_workers=2)
        cls_num = 100
    elif dataset == "flowers102":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_data = datasets.Flowers102(root = data_path, split = "train", download = True, transform = train_transform)
        train_loader = DataLoader(train_data, 512, shuffle = True, num_workers=4)
        test_data = datasets.Flowers102(root = data_path, split = "test", download = True, transform = test_transform)
        test_loader = DataLoader(test_data, 512, shuffle = False, num_workers=4)
        cls_num = 102
    elif dataset == "country211":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_data = datasets.Country211(root = data_path, split = "train", download = True, transform = train_transform)
        train_loader = DataLoader(train_data, 512, shuffle = True, num_workers=4)
        test_data = datasets.Country211(root = data_path, split = "test", download = True, transform = test_transform)
        test_loader = DataLoader(test_data, 512, shuffle = False, num_workers=4)
        cls_num = 211
    elif dataset == "oxfordpets":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_data = datasets.OxfordIIITPet(root = data_path, split = "trainval", download = True, transform = train_transform)
        train_loader = DataLoader(train_data, 512, shuffle = True, num_workers=4)
        test_data = datasets.OxfordIIITPet(root = data_path, split = "test", download = True, transform = test_transform)
        test_loader = DataLoader(test_data, 512, shuffle = False, num_workers=4)
        cls_num = 37
    return train_loader, test_loader, cls_num
