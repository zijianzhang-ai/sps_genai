import torch
from torchvision import datasets, transforms

def get_data_loader(data_dir, batch_size=32, train=True):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=True,
        transform=transform
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train
    )

    return loader

def get_fashion_data_loader(data_dir, batch_size=32, train=True):
    # The VAE practical pads 28x28 images to 32x32.
    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor()
    ])

    dataset = datasets.FashionMNIST(
        root=data_dir,
        train=train,
        download=True,
        transform=transform
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train
    )

    return loader

def get_assignment_data_loader(data_dir, batch_size=32, train=True):
    # Assignment 2 requires RGB images of size 64x64.
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=True,
        transform=transform
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train
    )

    return loader