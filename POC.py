import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from CNN_archi import AlexNet

def predict_image(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Reprise de la transformation de test (Resize 224x224 + Normalisation CIFAR-10)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Erreur lors de l'ouverture de l'image : {e}")
        return

    # Préparation du tensor (ajout de la dimension du batch : [1, 3, 224, 224])
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Chargement du modèle et des poids entraînés
    model = AlexNet(num_classes=10).to(device)
    try:
        model.load_state_dict(torch.load(r'AlexNet\models\alexnet_SDogs_epochs35.pth', map_location=device))
    except FileNotFoundError:
        print("Erreur : Le fichier 'alexnet_SDogs_epochs35.pth' est introuvable. Lancez d'abord training.py pour sauvegarder le modèle.")
        return

    model.eval()

    # Inférence sans calcul de gradient
    with torch.no_grad():
        outputs = model(input_tensor)
        # Conversion des logits bruts en pourcentages via Softmax
        probabilities = F.softmax(outputs, dim=1)[0] * 100

    # Étiquettes des classes CIFAR-10
    class_names = [
        'Avion', 'Automobile', 'Oiseau', 'Chat', 'Cerf', 
        'Chien', 'Grenouille', 'Cheval', 'Bateau', 'Camion'
    ]

    print(f"\nDistribution des probabilités pour : {image_path}")
    print("-" * 45)
    
    # Tri des prédictions par ordre décroissant
    prob_sorted, indices = torch.sort(probabilities, descending=True)
    
    for i in range(len(class_names)):
        idx = indices[i].item()
        prob = prob_sorted[i].item()
        print(f"{class_names[idx]:<12} : {prob:6.4f}%")
        
    print("-" * 45)
    print(f"-> Prédiction finale : {class_names[indices[0].item()]} ({prob_sorted[0].item():.4f}% de confiance)\n")

if __name__ == '__main__':
    path = r'AlexNet\test_pictures\poppydog.jpg'
    predict_image(path)