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
        
    # Same contract as inference
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = APTOSDataset(base_path / 'split_train.csv', base_path / 'train_images', transform)
    val_dataset = APTOSDataset(base_path / 'split_val.csv', base_path / 'train_images', transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 5)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    best_acc = 0.0
    epochs = 10
    
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
            
        # Eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = correct / total
        logging.info(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'models/dr_classification/best_model.pth')
            
    logging.info(f"Training complete. Best Val Acc: {best_acc:.4f}")
    
    # Save metadata
    metadata = {
        "model_name": "DR-SUGAR-Baseline-MobileNetV3",
        "architecture": "mobilenet_v3_small",
        "version": "1.0.0",
        "dataset": "APTOS 2019",
        "classes": 5,
        "class_mapping": {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"},
        "preprocessing_version": "v1-224-imagenet",
        "training_timestamp": datetime.datetime.now().isoformat(),
        "best_val_acc": best_acc
    }
    with open('models/dr_classification/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    train_model()
