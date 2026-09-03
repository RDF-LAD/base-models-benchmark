import torch
from sklearn.metrics import classification_report, accuracy_score
from CNN_archi import AlexNet
from data_importation import get_data_loaders

def main():
    # Configuration du GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Évaluation en cours sur : {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    # Récupération uniquement du dataloader de test
    _, testloader = get_data_loaders(batch_size=64)

    # Instanciation du modèle, chargement des poids entraînés et passage sur GPU
    model = AlexNet(num_classes=10).to(device)
    model.load_state_dict(torch.load(r'AlexNet\models\alexnet_cifar10_epochs50_PCA.pth'))
    
    # Passage en mode évaluation (désactive le Dropout)
    model.eval()

    all_preds = []
    all_labels = []

    print("Évaluation sur le jeu de test en cours...")
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Passe avant
            outputs = model(inputs)
            
            # Récupération de la classe prédite (indice du score le plus élevé)
            _, predicted = torch.max(outputs, 1)

            # Stockage des prédictions et des vraies étiquettes (sur le CPU)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Noms des classes CIFAR-10 pour un affichage lisible
    class_names = [
        'Avion', 'Automobile', 'Oiseau', 'Chat', 'Cerf', 
        'Chien', 'Grenouille', 'Cheval', 'Bateau', 'Camion'
    ]

    # Calcul des métriques globales et par classe
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nExactitude globale (Accuracy) : {accuracy * 100:.2f}%\n")

    print("Rapport de classification détaillé :")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

if __name__ == '__main__':
    main()