import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import random_split


def get_data_loaders(batch_size=64,img_size = 224,augm_max = True):
 
    # Transformations pour l'entraînement 
    pad= img_size // 8
    
    if augm_max :
        train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomCrop(img_size, padding=pad),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
        transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    else :
        train_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomCrop(img_size, padding=pad),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                ])
        
    # Transformations pour le test (strictement déterministes)
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    #Stratégie de chargement des datasets CIFAR-10
    full_trainset = torchvision.datasets.CIFAR10(root=r'C:\Users\User\Documents\projets vision\datasets', train=True, download=True, transform=train_transform)
    
    train_size = int(0.8 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = random_split(full_trainset, [train_size, val_size])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=r'C:\Users\User\Documents\projets vision\datasets', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, valloader, testloader
