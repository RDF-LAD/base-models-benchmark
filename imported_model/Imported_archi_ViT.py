from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import Trainer, TrainingArguments
from PIL import Image
import torch
import numpy as np

class PreTrainedViT:
    """
    Definit le modèle entrainable/testable basée sur ViT. (soit celui préentrainé par google soit un finetunné localement)
    """
    def __init__(self, num_classes,model_path=None):
        model_ckpt = model_path if model_path is not None else "google/vit-base-patch16-224"

        self.class_names = [
         'Avion', 'Automobile', 'Oiseau', 'Chat', 'Cerf', 
         'Chien', 'Grenouille', 'Cheval', 'Bateau', 'Camion'
     ]

        if torch.cuda.is_available():
              self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        print(f"Utilisation du périphérique : {self.device}")

        self.image_processor = AutoImageProcessor.from_pretrained(model_ckpt)

        if model_path is not None:
            self.model = AutoModelForImageClassification.from_pretrained(model_ckpt)
        else:
            self.model = AutoModelForImageClassification.from_pretrained(
                model_ckpt,
                num_labels=num_classes,
                ignore_mismatched_sizes=True  
            )
        self.model.to(self.device)

    def fine_tunning(self, train_ds, val_ds,n_epoch):
        """
        Entraîne le modèle à l'aide de l'API Trainer de Hugging Face 
        (qui gère automatiquement l'utilisation du GPU).
        """
        # Fonction optionnelle pour calculer l'accuracy pendant l'évaluation du Trainer
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            preds = np.argmax(predictions, axis=1)
            accuracy = (preds == labels).mean()
            return {"accuracy": float(accuracy)}

        training_args = TrainingArguments(
            output_dir="./vit-finetuned-custom",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            eval_strategy="epoch",  # Évaluation à chaque fin d'époque
            save_strategy="epoch",
            num_train_epochs=n_epoch,
            learning_rate=2e-5,  
            weight_decay=0.01,
            load_best_model_at_end=True, # Recharge le meilleur modèle basé sur la métrique d'évaluation
            metric_for_best_model="eval_loss",
            logging_steps=50, #touts les x etapes la metriques est affichée
            save_total_limit=2,
            fp16=torch.cuda.is_available(), 
            report_to="none" 
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        # Sauvegarde du modèle final et du processeur associé
        trainer.save_model("./FT_ViT")
        self.image_processor.save_pretrained("./FT_ViT")
        print("Modèle fine-tuné et sauvegardé dans './FT_ViT'")

    def test(self, testloader):
        """
        Évalue le modèle sur un DataLoader de test standard PyTorch en garantissant 
        que les calculs s'effectuent sur le GPU.
        """
        self.model.eval()
        correct_predictions = 0
        total_samples = 0
        
        print("Évaluation en cours sur le testloader...")
        with torch.no_grad():
            for images, labels in testloader:
                # Transfert explicite des données sur le GPU/CPU
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Passage en avant (forward pass) via pixel_values
                outputs = self.model(pixel_values=images)
                logits = outputs.logits
                
                # Récupération de la classe prédite
                _, predicted = torch.max(logits, dim=1)
                
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
        accuracy = (correct_predictions / total_samples) * 100
        print(f"Précision (Accuracy) sur le test set : {accuracy:.2f}% ({correct_predictions}/{total_samples})")
        return accuracy

    def inference(self, img_path):
        """
        Réalise une prédiction sur une image unique.
        """
        self.model.eval()

        image = Image.open(img_path).convert("RGB")
        inputs = self.image_processor(images=image, return_tensors="pt")
        
        # Déplacer les inputs sur le même appareil que le modèle
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            # Transformation des logits en probabilités via Softmax
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class_idx = int(torch.argmax(probabilities, dim=-1).item())
            confidence = probabilities[0, predicted_class_idx].item()
            print(f'La classe prédite est {self.class_names[predicted_class_idx]} avec une certitude de {np.round(confidence, 2) * 100}%')

        return predicted_class_idx, probabilities.cpu().numpy()