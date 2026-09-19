import torch
import torchvision
from torch.utils.data import random_split
from transformers import AutoImageProcessor


class HFTransform:
    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, image):
        return self.image_processor(image, return_tensors="pt")["pixel_values"][0]

def get_data_loaders(batch_size=64):
    """
    Permet l'importation ddu dataset CIFAR10 avec les transformations standards du modèle préentrainé sur ResNet
    """
    model_ckpt = "google/vit-base-patch16-224"
    image_processor = AutoImageProcessor.from_pretrained(model_ckpt)

    
    transform = HFTransform(image_processor)

    full_trainset = torchvision.datasets.CIFAR10(
        root=r'C:\Users\User\Documents\projets vision\datasets', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    train_size = int(0.8 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = random_split(full_trainset, [train_size, val_size])
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0)

    testset = torchvision.datasets.CIFAR10(
        root=r'C:\Users\User\Documents\projets vision\datasets', 
        train=False, 
        download=True, 
        transform=transform
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    return trainloader, valloader, testloader