import torch
from sklearn.metrics import accuracy_score, classification_report
from CNN_archi_AlexNet import AlexNet
from data_importation import get_data_loaders

def evaluate_model(model_path, testloader, device, model_name="Modèle"):
    print(f"\n--- Évaluation de : {model_name} ({model_path}) ---")
    
    model = AlexNet(num_classes=10).to(device)
    try:
        # Chargement du checkpoint complet
        checkpoint = torch.load(model_path, map_location=device)
        
        # Gestion du cas où le fichier est un dictionnaire d'entraînement complet
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                # Si le dictionnaire contient directement les poids mais sous d'autres clés
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Chargement sécurisé des poids
        model.load_state_dict(state_dict, strict=False)
        
    except Exception as e:
        print(f"Erreur lors du chargement de {model_path} : {e}")
        return None, None

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Exactitude globale ({model_name}) : {accuracy * 100:.2f}%\n")
    return all_labels, all_preds

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Comparaison en cours sur : {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    _, testloader = get_data_loaders(batch_size=64)

    class_names = [
        'Avion', 'Automobile', 'Oiseau', 'Chat', 'Cerf', 
        'Chien', 'Grenouille', 'Cheval', 'Bateau', 'Camion'
    ]

    # 1. Évaluation de ton modèle local
    path_local = r'AlexNet\models\alexnet_cifar10_epochs10.pth'
    labels_ref, preds_local = evaluate_model(path_local, testloader, device, model_name="Mon Modèle Local")

    # 2. Évaluation du modèle externe (téléchargé depuis un dépôt tiers)
    path_external = r'AlexNet\models\EXTERNAL_alexnet_cifar10.pth'
    _, preds_external = evaluate_model(path_external, testloader, device, model_name="Modèle Externe Référence")

    # Affichage des rapports détaillés si les deux modèles ont pu être chargés
    if preds_local is not None and preds_external is not None:
        print("=" * 50)
        print("Rapport détaillé : Mon Modèle Local")
        print(classification_report(labels_ref, preds_local, target_names=class_names, digits=4))
        
        print("=" * 50)
        print("Rapport détaillé : Modèle Externe Référence")
        print(classification_report(labels_ref, preds_external, target_names=class_names, digits=4))

if __name__ == '__main__':
    main()