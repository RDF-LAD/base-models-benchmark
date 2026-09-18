import torch
import torchvision
from torch.utils.data import random_split
from transformers import AutoImageProcessor

def get_data_loaders(batch_size=64):
    
    model_ckpt = "google/vit-base-patch16-224"
    image_processor = AutoImageProcessor.from_pretrained(model_ckpt)
    
    def hf_transform(image):
        return image_processor(image, return_tensors="pt")["pixel_values"].squeeze(0)
    
    transform = torchvision.transforms.Lambda(lambda img: hf_transform(img))

    full_trainset = torchvision.datasets.CIFAR10(
        root=r'C:\Users\User\Documents\projets vision\datasets', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    train_size = int(0.8 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = random_split(full_trainset, [train_size, val_size])
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root=r'C:\Users\User\Documents\projets vision\datasets', 
        train=False, 
        download=True, 
        transform=transform
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, valloader, testloader