import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(dataset_name='cifar10', data_dir='./data', batch_size=128, img_size=32, num_workers=4):
    
    os.makedirs(data_dir, exist_ok=True)
    
    if dataset_name.lower() == 'mnist':
        in_channels = 1
        num_classes = 10
        mean, std = [0.1307], [0.3081]
    elif dataset_name.lower() == 'cifar10':
        in_channels = 3
        num_classes = 10
        mean, std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
    elif dataset_name.lower() == 'cifar100':
        in_channels = 3
        num_classes = 100
        mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported!")

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip() if dataset_name.lower() in ['cifar10', 'cifar100'] else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if dataset_name.lower() == 'mnist':
        train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=test_transform)
    elif dataset_name.lower() == 'cifar10':
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    elif dataset_name.lower() == 'cifar100':
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader, num_classes, in_channels