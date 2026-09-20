import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import pandas as pd
from PIL import Image
from pathlib import Path
import json
import logging
import datetime
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

logging.basicConfig(level=logging.INFO)

class IDRiDDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform
        
        # IDRiD columns: 'Image name', 'Retinopathy grade'
        # Fallback if different
        self.img_col = self.df.columns[0]
        self.label_col = self.df.columns[1]
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx][self.img_col]
        img_path = self.img_dir / f"{img_name}.jpg"
        image = Image.open(img_path).convert('RGB')
        label = int(self.df.iloc[idx][self.label_col])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_class_weights(df, label_col):
    class_counts = df[label_col].value_counts().sort_index()
    total_samples = len(df)
    # class_weight = total_samples / (num_classes * count)
    weights = []
    num_classes = 5
    for i in range(num_classes):
        count = class_counts.get(i, 0)
        weight = total_samples / (num_classes * count) if count > 0 else 0
        weights.append(weight)
    return torch.FloatTensor(weights)

def train_model():
    base_path = Path('data/idrid/B. Disease Grading')
    gt_path = base_path / '2. Groundtruths'
    train_img_dir = base_path / '1. Original Images' / 'a. Training Set'
    test_img_dir = base_path / '1. Original Images' / 'b. Testing Set'
    
    train_csv = gt_path / 'split_train.csv'
    val_csv = gt_path / 'split_val.csv'
    test_csv = gt_path / 'split_test.csv'
    
    if not train_csv.exists():
        logging.error("Splits not found. Run prepare_idrid_dataset.py first.")
        return
        
    # Model save path
    save_dir = Path('models/idrid')
    save_dir.mkdir(parents=True, exist_ok=True)
        
    # Preprocessing
    # Using exactly the requested transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = IDRiDDataset(train_csv, train_img_dir, train_transform)
    val_dataset = IDRiDDataset(val_csv, train_img_dir, eval_transform)
    test_dataset = IDRiDDataset(test_csv, test_img_dir, eval_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info(f"Using device: {device}")
    
    # Class weights
    class_weights = get_class_weights(train_dataset.df, train_dataset.label_col).to(device)
    logging.info(f"Class weights: {class_weights}")
    
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 5)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    best_val_f1 = 0.0
    epochs = 15 # Allow enough epochs for IDRiD since dataset is small
    
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
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_dir / 'best_model.pth')
            
    logging.info(f"Training complete. Best Val F1: {best_val_f1:.4f}")
    
    # --- TEST EVALUATION ---
    logging.info("Evaluating on HELD-OUT TEST SET...")
    model.load_state_dict(torch.load(save_dir / 'best_model.pth', map_location=device))
    model.eval()
    
    test_preds = []
    test_labels = []
    test_probs = []
    test_img_ids = test_dataset.df[test_dataset.img_col].tolist()
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
            
    test_acc = accuracy_score(test_labels, test_preds)
    test_macro_prec = precision_score(test_labels, test_preds, average='macro', zero_division=0)
    test_macro_rec = recall_score(test_labels, test_preds, average='macro', zero_division=0)
    test_macro_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)
    test_weighted_f1 = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
    
    cm = confusion_matrix(test_labels, test_preds, labels=[0, 1, 2, 3, 4])
    
    # Per-class metrics
    per_class_prec = precision_score(test_labels, test_preds, average=None, zero_division=0)
    per_class_rec = recall_score(test_labels, test_preds, average=None, zero_division=0)
    per_class_f1 = f1_score(test_labels, test_preds, average=None, zero_division=0)
    
    # Referable DR calculation (severity >= 2)
    ref_labels = [1 if x >= 2 else 0 for x in test_labels]
    ref_preds = [1 if x >= 2 else 0 for x in test_preds]
    
    ref_cm = confusion_matrix(ref_labels, ref_preds, labels=[0, 1])
    TN, FP, FN, TP = ref_cm.ravel() if ref_cm.size == 4 else (0,0,0,0)
    
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    ref_prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    ref_f1 = 2 * (ref_prec * sensitivity) / (ref_prec + sensitivity) if (ref_prec + sensitivity) > 0 else 0
    
    logging.info(f"Test Acc: {test_acc:.4f} | Test F1: {test_macro_f1:.4f}")
    logging.info(f"Referable DR -> TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    logging.info(f"Sensitivity: {sensitivity:.4f} | Specificity: {specificity:.4f}")
    
    # Generate test_predictions.csv
    pred_data = []
    for i in range(len(test_labels)):
        p = test_probs[i]
        pred_data.append({
            'image_id': test_img_ids[i],
            'true_grade': int(test_labels[i]),
            'predicted_grade': int(test_preds[i]),
            'prob_0': float(p[0]),
            'prob_1': float(p[1]),
            'prob_2': float(p[2]),
            'prob_3': float(p[3]),
            'prob_4': float(p[4]),
            'confidence': float(np.max(p)),
            'referable_true': int(ref_labels[i]),
            'referable_pred': int(ref_preds[i])
        })
    pd.DataFrame(pred_data).to_csv(save_dir / 'test_predictions.csv', index=False)
    
    # Save metadata
    metadata = {
        "model_name": "DR-SUGAR-IDRiD-MobileNetV3",
        "architecture": "mobilenet_v3_small",
        "version": "1.0.0",
        "dataset": "IDRiD B. Disease Grading",
        "dataset_samples": len(train_dataset) + len(val_dataset) + len(test_dataset),
        "classes": 5,
        "class_mapping": {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"},
        "preprocessing_version": "v1-224-imagenet-aug",
        "random_seed": 42,
        "imbalance_strategy": "Class-Weighted Cross Entropy",
        "training_config": {
            "epochs": epochs,
            "optimizer": "Adam",
            "lr": 1e-4,
            "batch_size": 16
        },
        "checkpoint_path": str(save_dir / 'best_model.pth'),
        "evaluation_split": "split_test.csv",
        "metrics": {
            "test_accuracy": float(test_acc),
            "test_macro_f1": float(test_macro_f1),
            "test_weighted_f1": float(test_weighted_f1),
            "test_macro_precision": float(test_macro_prec),
            "test_macro_recall": float(test_macro_rec),
            "confusion_matrix": cm.tolist(),
            "per_class": {
                "precision": per_class_prec.tolist(),
                "recall": per_class_rec.tolist(),
                "f1": per_class_f1.tolist()
            },
            "referable_dr": {
                "TP": int(TP),
                "TN": int(TN),
                "FP": int(FP),
                "FN": int(FN),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "precision": float(ref_prec),
                "f1": float(ref_f1)
            },
            "calibration": {
                "implemented": False,
                "note": "Calibration not implemented. Raw softmax probabilities provided."
            }
        },
        "training_timestamp": datetime.datetime.now().isoformat()
    }
    with open(save_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    train_model()
