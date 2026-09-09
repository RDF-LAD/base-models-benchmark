import os

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from CNN_archi_AlexNet import AlexNet
from CNN_archi_ResNet import ResNet
from CNN_archi_ViT import VisionTransformer

def predict_image(image_path,dataset='cifar10',model_type='AlexNet',epochs=35):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if dataset == 'cifar10':
        # Reprise de la transformation de test (Resize 224x224 + Normalisation CIFAR-10)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])

    else : 
        transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.476, 0.451, 0.391), std=(0.258, 0.252, 0.257))
    ])
        
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Erreur lors de l'ouverture de l'image : {e}")
        return

    # Préparation du tensor (ajout de la dimension du batch : [1, 3, 224, 224])
    input_tensor = transform(image).unsqueeze(0).to(device)

    if dataset == 'cifar10':
    
            n_class = 10
            class_names = [
             'Avion', 'Automobile', 'Oiseau', 'Chat', 'Cerf', 
             'Chien', 'Grenouille', 'Cheval', 'Bateau', 'Camion'
         ]
    else :

            n_class = 120
            data_dir = r'C:\Users\User\Documents\projets vision\datasets\Stanford_dogs\Images\Images'
            class_names = sorted([d.split('-', 1)[1] for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    # Chargement du modèle et des poids entraînés

    if model_type == 'AlexNet': 
            model = AlexNet(num_classes=n_class).to(device) # 10 classes for Cifar-10 / 120 for stanford dogs
    
    elif model_type == 'ResNet' :
            model = ResNet(num_classes=n_class).to(device) # 10 classes for Cifar-10 / 120 for stanford dogs

    elif model_type == 'ViT':
            model = VisionTransformer(num_classes=n_class).to(device)            
    
    model.load_state_dict(torch.load(r'C:\Users\User\Documents\projets vision\AlexNet\models//'+model_type+'_'+dataset+'_'+str(epochs)+'.pth'))

    model.eval()

    # Inférence sans calcul de gradient
    with torch.no_grad():
        outputs = model(input_tensor)
        # Conversion des logits bruts en pourcentages via Softmax
        probabilities = F.softmax(outputs, dim=1)[0] * 100

    print(f"\nDistribution des probabilités pour : {image_path}")
    print("-" * 45)
    
    prob_sorted, indices = torch.sort(probabilities, descending=True)
    
    for i in range(len(class_names)):
        idx = indices[i].item()
        prob = prob_sorted[i].item()
        print(f"{class_names[idx]:<12} : {prob:6.4f}%")
        
    print("-" * 45)
    print(f"-> Prédiction finale : {class_names[indices[0].item()]} ({prob_sorted[0].item():.4f}% de confiance)\n")

if __name__ == '__main__':
    path = r'C:\Users\User\Documents\projets vision\AlexNet\test_pictures\chien3.jpeg'
    predict_image(path,dataset='cifar10',model_type='ResNet',epochs=50)
