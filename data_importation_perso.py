import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

def compute_mean_and_std(dataloader):
    """
    Permet de calculer mean et std pour normaliser le dataset
    """
    pixel_sum = torch.zeros(3)
    pixel_squared_sum = torch.zeros(3)
    num_pixels = 0

    for images, _ in dataloader:
        # images shape: [batch_size, 3, height, width]
        b, c, h, w = images.shape
        num_pixels += b * h * w
        
        # Somme sur les dimensions batch, height et width
        pixel_sum += images.sum(dim=[0, 2, 3])
        pixel_squared_sum += (images ** 2).sum(dim=[0, 2, 3])
    
    mean = pixel_sum / num_pixels
    std = torch.sqrt((pixel_squared_sum / num_pixels) - (mean ** 2))

    return mean.tolist(), std.tolist()

class StanfordDogsFolderDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

def get_stanford_dogs_split_loaders(root_dir=r'C:\Users\User\Documents\projets vision\datasets\Stanford_dogs\Images\Images', batch_size=32, train_ratio=0.8):

    # Récupérer tous les sous-dossiers (les races) et les trier alphabétiquement
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    train_paths, train_labels = [], []
    test_paths, test_labels = [], []

    # Parcourir chaque race pour faire le split 80/20
    for cls_name in classes:
        cls_dir = os.path.join(root_dir, cls_name)
        images = [os.path.join(cls_dir, img) for img in os.listdir(cls_dir) if img.lower().endswith(('jpg', 'jpeg', 'png'))]
        
        # Mélanger aléatoirement les images de cette classe
        random.seed(42) # Assure la reproductibilité du split
        random.shuffle(images)
        
        split_idx = int(len(images) * train_ratio)
        
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        label_idx = class_to_idx[cls_name]
        
        train_paths.extend(train_images)
        train_labels.extend([label_idx] * len(train_images))
        
        test_paths.extend(test_images)
        test_labels.extend([label_idx] * len(test_images))

    # Pipelines de transformation distincts pour train et test
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=24),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.8, 1.2)),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
                transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.476, 0.451, 0.391), std=(0.258, 0.252, 0.257))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.476, 0.451, 0.391), std=(0.258, 0.252, 0.257))
    ])

    # Instanciation des datasets PyTorch
    trainset = StanfordDogsFolderDataset(train_paths, train_labels, transform=train_transform)
    testset = StanfordDogsFolderDataset(test_paths, test_labels, transform=test_transform)

    # Création des DataLoaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Nombre de classes détectées : {len(classes)}")
    print(f"Total images d'entraînement : {len(trainset)}")
    print(f"Total images de test : {len(testset)}")

    return trainloader, testloader

if __name__ == '__main__':
     train_loader, test_loader = get_stanford_dogs_split_loaders()
#      print(compute_mean_and_std(train_loader))