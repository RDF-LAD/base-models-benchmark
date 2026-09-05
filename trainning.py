import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from CNN_archi_AlexNet import AlexNet
from data_importation import get_data_loaders
from data_importation_perso import get_stanford_dogs_split_loaders
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Entraînement en cours sur : {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    # Récupération des dataloaders
    #trainloader, testloader = get_data_loaders(batch_size=64)
    trainloader, valloader,_ = get_stanford_dogs_split_loaders(batch_size=64)

    # Instanciation du modèle et passage sur GPU
    model = AlexNet(num_classes=120).to(device) # 10 classes for Cifar-10 / 120 for stanford dogs

    # Fonction de perte et optimisateur
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',     # On cherche à minimiser la loss
    factor=0.1,     # Divise le LR par 10 (new_lr = lr * 0.1)
    patience=5,     # Nombre d'époques à attendre sans amélioration avant d'agir
    threshold=1e-4  # Seuil minimal pour considérer qu'il y a une amélioration
)

    # Listes pour stocker l'historique des pertes
    train_losses = []
    val_losses = []

    # Nombre d'époques      
    num_epochs = 35 # after 40 epochs it starts to overfit

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
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_test_loss += loss.item()
                
        epoch_test_loss = running_test_loss / len(valloader)
        val_losses.append(epoch_test_loss)
        # Mise à jour du scheduler basé sur la loss de test
        scheduler.step(epoch_test_loss)

        # Affichage des résultats de l'époque
        print(f"Epochs [{epoch+1}/{num_epochs}] | Train Loss: {epoch_train_loss:.4f} | Test Loss: {epoch_test_loss:.4f}")

    print("Entraînement terminé avec succès !")

    # --- 3. AFFICHAGE DES COURBES ---
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o', color='blue')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='o', color='orange')
    plt.title('Loss Evolution (Training vs Validation)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (CrossEntropy)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # À la fin de training.py, après l'entraînement
    torch.save(model.state_dict(), 'alexnet_cifar10.pth')
    print("Modèle sauvegardé avec succès sous 'alexnet_cifar10.pth'")

if __name__ == '__main__':
    main()