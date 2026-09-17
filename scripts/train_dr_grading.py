import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
from PIL import Image
from pathlib import Path
import json
import logging
import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

logging.basicConfig(level=logging.INFO)

class APTOSDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['id_code']
        img_path = self.img_dir / f"{img_name}.png"
        image = Image.open(img_path).convert('RGB')
        label = self.df.iloc[idx]['diagnosis']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def train_model():
    base_path = Path('data/aptos2019')
    if not (base_path / 'split_train.csv').exists():
        logging.error("Splits not found. Run prepare_dr_dataset.py first.")
        return
        
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = APTOSDataset(base_path / 'split_train.csv', base_path / 'train_images', transform)
    val_dataset = APTOSDataset(base_path / 'split_val.csv', base_path / 'train_images', transform)
    test_dataset = APTOSDataset(base_path / 'split_test.csv', base_path / 'train_images', transform)
    
    # Very small batch size because it's CPU training and might be memory intensive
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 5)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    best_val_f1 = 0.0
    epochs = 3 # Kept small for CPU baseline.
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        train_loss = running_loss / len(train_loader)
        
        # Eval on Val
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_loss /= len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        logging.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        # Save best based on Macro F1 to handle imbalance slightly better
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'models/dr_classification/best_model.pth')
            
    logging.info(f"Training complete. Best Val F1: {best_val_f1:.4f}")
    
    # --- TEST EVALUATION ---
    logging.info("Evaluating on HELD-OUT TEST SET...")
    model.load_state_dict(torch.load('models/dr_classification/best_model.pth'))
    model.eval()
    
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            
    test_acc = accuracy_score(test_labels, test_preds)
    test_macro_prec = precision_score(test_labels, test_preds, average='macro', zero_division=0)
    test_macro_rec = recall_score(test_labels, test_preds, average='macro', zero_division=0)
    test_macro_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)
    
    cm = confusion_matrix(test_labels, test_preds, labels=[0, 1, 2, 3, 4])
    
    # Referable DR calculation (severity >= 2)
    # 0,1 -> 0 (Negative)
    # 2,3,4 -> 1 (Positive)
    ref_labels = [1 if x >= 2 else 0 for x in test_labels]
    ref_preds = [1 if x >= 2 else 0 for x in test_preds]
    
    # Binary confusion matrix values
    ref_cm = confusion_matrix(ref_labels, ref_preds, labels=[0, 1])
    TN, FP, FN, TP = ref_cm.ravel() if ref_cm.size == 4 else (0,0,0,0)
    
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    ref_prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    ref_f1 = 2 * (ref_prec * sensitivity) / (ref_prec + sensitivity) if (ref_prec + sensitivity) > 0 else 0
    
    logging.info(f"Test Acc: {test_acc:.4f} | Test F1: {test_macro_f1:.4f}")
    logging.info(f"Referable DR -> TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    logging.info(f"Sensitivity: {sensitivity:.4f} | Specificity: {specificity:.4f}")
    
    # Save metadata
    metadata = {
        "model_name": "DR-SUGAR-Baseline-MobileNetV3",
        "architecture": "mobilenet_v3_small",
        "parameters": "2.54M",
        "version": "1.0.0",
        "dataset": "APTOS 2019",
        "dataset_samples": len(train_dataset) + len(val_dataset) + len(test_dataset),
        "classes": 5,
        "class_mapping": {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"},
        "preprocessing_version": "v1-224-imagenet",
        "random_seed": 42,
        "training_config": {
            "epochs": epochs,
            "optimizer": "Adam",
            "lr": 1e-4,
            "batch_size": 16
        },
        "checkpoint_path": "models/dr_classification/best_model.pth",
        "evaluation_split": "split_test.csv",
        "metrics": {
            "test_accuracy": float(test_acc),
            "test_macro_f1": float(test_macro_f1),
            "test_macro_precision": float(test_macro_prec),
            "test_macro_recall": float(test_macro_rec),
            "confusion_matrix": cm.tolist(),
            "referable_dr": {
                "TP": int(TP),
                "TN": int(TN),
                "FP": int(FP),
                "FN": int(FN),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "precision": float(ref_prec),
                "f1": float(ref_f1)
            }
        },
        "training_timestamp": datetime.datetime.now().isoformat()
    }
    with open('models/dr_classification/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    train_model()
