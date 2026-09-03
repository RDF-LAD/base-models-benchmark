import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from CNN_archi import AlexNet
from data_importation import get_data_loaders
import matplotlib.pyplot as plt

def main():
    # Configuration du GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Entraînement en cours sur : {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    # Récupération des dataloaders
    trainloader, testloader = get_data_loaders(batch_size=64)

    # Instanciation du modèle et passage sur GPU
    model = AlexNet(num_classes=10).to(device)

    # Fonction de perte et optimisateur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

    # Listes pour stocker l'historique des pertes
    train_losses = []
    test_losses = []

    # Nombre d'époques (tu pourras l'augmenter, ex: 20 ou 50 pour observer l'overfitting)
    num_epochs = 50

    print("\n--- Début de l'entraînement ---")
    for epoch in range(num_epochs):
        
        # --- 1. PHASE D'ENTRAÎNEMENT ---
        model.train()
        running_train_loss = 0.0
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            
        epoch_train_loss = running_train_loss / len(trainloader)
        train_losses.append(epoch_train_loss)

        # --- 2. PHASE D'ÉVALUATION (TEST) ---
        model.eval() # Désactive le dropout/batchnorm spécifiques à l'entraînement
        running_test_loss = 0.0
        
        with torch.no_grad(): # Pas besoin de calculer les gradients pour le test (économise la mémoire)
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_test_loss += loss.item()
                
        epoch_test_loss = running_test_loss / len(testloader)
        test_losses.append(epoch_test_loss)

        # Affichage des résultats de l'époque
        print(f"Époque [{epoch+1}/{num_epochs}] | Train Loss: {epoch_train_loss:.4f} | Test Loss: {epoch_test_loss:.4f}")

    print("Entraînement terminé avec succès !")

    # --- 3. AFFICHAGE DES COURBES ---
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o', color='blue')
    plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss', marker='o', color='orange')
    plt.title('Évolution de la Loss (Entraînement vs Test)')
    plt.xlabel('Époques')
    plt.ylabel('Loss (CrossEntropy)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # À la fin de training.py, après l'entraînement
    torch.save(model.state_dict(), 'alexnet_cifar10.pth')
    print("Modèle sauvegardé avec succès sous 'alexnet_cifar10.pth'")

if __name__ == '__main__':
    main()