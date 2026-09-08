import os

import torch
from sklearn.metrics import classification_report, accuracy_score
from CNN_archi_AlexNet import AlexNet
from CNN_archi_ResNet import ResNet
from data_importation import get_data_loaders
from data_importation_perso import get_stanford_dogs_split_loaders

def top_5_accuracy(y_true, y_pred_top5):
    """
    Calcule l'exactitude top-5.
    y_true : liste des vraies étiquettes
    y_pred_top5 : liste des 5 classes prédites pour chaque échantillon
    """
    correct = 0
    for true_label, top5 in zip(y_true, y_pred_top5):
        if true_label in top5:
            correct += 1
    return correct / len(y_true)

def main(dataset = 'cifar10',model_name = 'AlexNet',epochs = 35):
    # Configuration du GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Évaluation en cours sur : {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    # Récupération uniquement du dataloader de test
    if dataset == 'cifar10':
        _,_, testloader = get_data_loaders(batch_size=64)
        n_class = 10
        class_names = [
         'Avion', 'Automobile', 'Oiseau', 'Chat', 'Cerf', 
         'Chien', 'Grenouille', 'Cheval', 'Bateau', 'Camion'
     ]
    else :
        _,_, testloader = get_stanford_dogs_split_loaders(batch_size=64)
        n_class = 120
        data_dir = r'C:\Users\User\Documents\projets vision\datasets\Stanford_dogs\Images\Images'
        class_names = sorted([d.split('-', 1)[1] for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    # Instanciation du modèle, chargement des poids entraînés et passage sur GPU
    if model_name == 'AlexNet': 
        model = AlexNet(num_classes=n_class).to(device) # 10 classes for Cifar-10 / 120 for stanford dogs

    else :
        model = ResNet(num_classes=n_class).to(device) # 10 classes for Cifar-10 / 120 for stanford dogs

    model.load_state_dict(torch.load(r'C:\Users\User\Documents\projets vision\AlexNet\models//'+model_name+'_'+dataset+'_'+str(epochs)+'.pth'))
    # Passage en mode évaluation (désactive le Dropout)
    model.eval()

    all_preds = []
    all_labels = []

    all_top5_preds = []

    print("Évaluation sur le jeu de test en cours...")
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Passe avant
            outputs = model(inputs)
            
            # Récupération de la classe prédite (indice du score le plus élevé)
            _, predicted = torch.max(outputs, 1)

            #Récupération des top 5 classes prédits
            if dataset != 'cifar10' :
                _,top5_pred = torch.topk(outputs, k=5, dim=1)
                all_top5_preds.extend(top5_pred.cpu().numpy())

            # Stockage des prédictions et des vraies étiquettes (sur le CPU)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            


    # Calcul des métriques globales et par classe
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nExactitude globale (Accuracy) : {accuracy * 100:.2f}%\n")

        # Calcul de l'exactitude top-5
    if dataset != 'cifar10' :
        top5_accuracy = top_5_accuracy(all_labels, all_top5_preds)
        print(f"\nExactitude top-5 : {top5_accuracy * 100:.2f}%\n")


    print("Rapport de classification détaillé :")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))


if __name__ == '__main__':
    main(dataset = 'Sanford',model_name='ResNet',epochs=50)