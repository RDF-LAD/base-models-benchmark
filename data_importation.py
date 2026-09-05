import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import random_split

class PCAColorAugmentation:
    """
    Implémentation de l'augmentation de couleur par ACP (PCA) d'AlexNet.
    Modifie les intensités RGB selon les composantes principales du dataset.
    """
    def __init__(self, eigen_values, eigen_vectors, std=0.1):
        self.eigen_values = eigen_values
        self.eigen_vectors = eigen_vectors
        self.std = std

    def __call__(self, tensor):
        # tensor attendu au format PyTorch [C, H, W] normalisé ou en [0, 1]
        if self.std == 0:
            return tensor
        
        # Génération des variables aléatoires alpha (gaussienne de moyenne 0, écart-type 0.1)
        alpha = np.random.normal(0, self.std, size=3)
        
        # Calcul du delta : [p1, p2, p3] * [alpha1 * lambda1, alpha2 * lambda2, alpha3 * lambda3]^T
        # self.eigen_vectors est de dimension (3, 3), eigen_values de dimension (3,)
        delta = np.dot(self.eigen_vectors, self.eigen_values * alpha)
        
        # Application du delta sur les canaux [C, H, W]
        tensor = tensor.clone()
        for i in range(3):
            tensor[i] += float(delta[i])
            
        return torch.clamp(tensor, 0.0, 1.0)

def get_data_loaders(batch_size=64):
    # Valeurs PCA approximatives calculées sur l'ensemble CIFAR-10 (canaux RGB)
    # Valeurs propres (Eigenvalues)
    cifar_eigen_values = np.array([0.2175, 0.0188, 0.0045], dtype=np.float32)
    # Vecteurs propres (Eigenvectors - matrice 3x3)
    cifar_eigen_vectors = np.array([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203]
    ], dtype=np.float32)

    # Transformations pour l'entraînement (incluant le PCA Color et les bruits géométriques/flou)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=24),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
        transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        #PCAColorAugmentation(cifar_eigen_values, cifar_eigen_vectors, std=0.1), # Appliqué ici sur le tenseur brut [0, 1]
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # Transformations pour le test (strictement déterministes)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    #Stratégie de chargement des datasets CIFAR-10
    full_trainset = torchvision.datasets.CIFAR10(root=r'C:\Users\User\Documents\projets vision\datasets', train=True, download=True, transform=train_transform)
    # 2. Le diviser en Train (ex: 45 000 images) et Validation (ex: 5 000 images)
    train_size = int(0.9 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = random_split(full_trainset, [train_size, val_size])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=r'C:\Users\User\Documents\projets vision\datasets', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, valloader, testloader