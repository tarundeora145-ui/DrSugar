"""
DR-SUGAR — IDRiD Disease Grading V2 Training
=============================================
Changes from V1:
  - WeightedRandomSampler on the training loader (replaces class-weighted loss)
  - Plain CrossEntropyLoss (no class weights)
  - 30 epochs maximum
  - ReduceLROnPlateau scheduler (patience=4, factor=0.5, mode='max')
  - Early stopping (patience=8) based on validation Macro F1

Identical to V1:
  - MobileNetV3-Small, ImageNet initialisation
  - 224×224 input, RGB, ImageNet normalisation
  - Same augmentations (h-flip, rotation ±10°, mild colour jitter)
  - seed 42
  - Same 80/20 split (split_train.csv / split_val.csv)
  - Held-out test set ONLY evaluated once after training is complete
  - Validation and test loaders: no weighted sampling

Research/prototype only. Not clinically validated.
"""

import os
import sys
import json
import logging
import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO, format='%(levelname)s:root:%(message)s')

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

NUM_CLASSES = 5
BATCH_SIZE = 16
MAX_EPOCHS = 30
LR = 1e-4
EARLY_STOP_PATIENCE = 8     # epochs without val Macro F1 improvement
SCHEDULER_PATIENCE = 4       # ReduceLROnPlateau patience
SCHEDULER_FACTOR = 0.5

BASE_PATH = Path('data/idrid/B. Disease Grading')
GT_PATH = BASE_PATH / '2. Groundtruths'
TRAIN_IMG_DIR = BASE_PATH / '1. Original Images' / 'a. Training Set'
TEST_IMG_DIR = BASE_PATH / '1. Original Images' / 'b. Testing Set'
TRAIN_CSV = GT_PATH / 'split_train.csv'
VAL_CSV = GT_PATH / 'split_val.csv'
TEST_CSV = GT_PATH / 'split_test.csv'

SAVE_DIR = Path('models/idrid/v2')


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class IDRiDDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform
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


# ---------------------------------------------------------------------------
# Weighted sampler — built from training labels so every epoch draws samples
# proportional to the inverse class frequency.
# ---------------------------------------------------------------------------

def build_weighted_sampler(dataset: IDRiDDataset) -> WeightedRandomSampler:
    labels = dataset.df[dataset.label_col].astype(int).tolist()
    class_counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    # Avoid division by zero for absent classes
    class_counts = np.where(class_counts == 0, 1, class_counts)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[l] for l in labels]
    sample_weights_tensor = torch.DoubleTensor(sample_weights)
    # num_samples = len(labels): draw same number of samples per epoch as the dataset size
    sampler = WeightedRandomSampler(sample_weights_tensor, num_samples=len(labels), replacement=True)
    return sampler


# ---------------------------------------------------------------------------
# Evaluation helper — called on val and test sets
# ---------------------------------------------------------------------------

def evaluate(model, loader, device, criterion=None):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs, all_img_ids = [], [], [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if criterion is not None:
                total_loss += criterion(outputs, labels).item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / max(len(loader), 1)
    acc = accuracy_score(all_labels, all_preds)
    macro_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    per_prec = precision_score(all_labels, all_preds, average=None, zero_division=0, labels=list(range(NUM_CLASSES)))
    per_rec = recall_score(all_labels, all_preds, average=None, zero_division=0, labels=list(range(NUM_CLASSES)))
    per_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0, labels=list(range(NUM_CLASSES)))

    ref_true = [1 if x >= 2 else 0 for x in all_labels]
    ref_pred = [1 if x >= 2 else 0 for x in all_preds]
    ref_cm = confusion_matrix(ref_true, ref_pred, labels=[0, 1])
    if ref_cm.size == 4:
        TN, FP, FN, TP = ref_cm.ravel()
    else:
        TN, FP, FN, TP = 0, 0, 0, 0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    ref_prec_val = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    ref_f1 = 2 * (ref_prec_val * sensitivity) / (ref_prec_val + sensitivity) if (ref_prec_val + sensitivity) > 0 else 0.0

    # ROC-AUC (only meaningful if both classes represented in ground truth)
    ref_roc_auc = None
    ref_probs = [p[2] + p[3] + p[4] for p in all_probs]
    if len(set(ref_true)) > 1:
        ref_roc_auc = roc_auc_score(ref_true, ref_probs)

    return {
        'loss': avg_loss,
        'accuracy': acc,
        'macro_precision': macro_prec,
        'macro_recall': macro_rec,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'confusion_matrix': cm,
        'per_class_precision': per_prec,
        'per_class_recall': per_rec,
        'per_class_f1': per_f1,
        'referable_dr': {
            'TP': int(TP), 'TN': int(TN), 'FP': int(FP), 'FN': int(FN),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(ref_prec_val),
            'f1': float(ref_f1),
            'roc_auc': float(ref_roc_auc) if ref_roc_auc is not None else None,
        },
        'preds': all_preds,
        'labels': all_labels,
        'probs': all_probs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_model():
    for csv_path in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
        if not csv_path.exists():
            logging.error(f"Split CSV not found: {csv_path}. Run prepare_idrid_dataset.py first.")
            return

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = IDRiDDataset(TRAIN_CSV, TRAIN_IMG_DIR, train_transform)
    val_dataset = IDRiDDataset(VAL_CSV, TRAIN_IMG_DIR, eval_transform)
    test_dataset = IDRiDDataset(TEST_CSV, TEST_IMG_DIR, eval_transform)

    # Log class distribution in training split
    train_labels = train_dataset.df[train_dataset.label_col].astype(int).tolist()
    counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    logging.info(f"Training split class distribution: {dict(enumerate(counts.tolist()))}")

    # WeightedRandomSampler — training only
    sampler = build_weighted_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
    model = model.to(device)

    # Plain CrossEntropyLoss (no class weights) — balance is handled by sampler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ReduceLROnPlateau: reduce LR when val Macro F1 stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
    )

    best_val_f1 = 0.0
    epochs_without_improvement = 0
    val_metrics_history = []

    logging.info("=== V2 Training Start ===")
    logging.info(f"Strategy: WeightedRandomSampler + CrossEntropyLoss (no class weights)")
    logging.info(f"Scheduler: ReduceLROnPlateau (patience={SCHEDULER_PATIENCE}, factor={SCHEDULER_FACTOR})")
    logging.info(f"Early stopping: patience={EARLY_STOP_PATIENCE} (metric: Val Macro F1)")

    for epoch in range(MAX_EPOCHS):
        # --- Train ---
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

        # --- Val ---
        val_result = evaluate(model, val_loader, device, criterion)
        val_loss = val_result['loss']
        val_acc = val_result['accuracy']
        val_f1 = val_result['macro_f1']

        scheduler.step(val_f1)

        logging.info(
            f"Epoch {epoch+1}/{MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        val_metrics_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_macro_f1': val_f1,
            'val_macro_precision': val_result['macro_precision'],
            'val_macro_recall': val_result['macro_recall'],
            'val_weighted_f1': val_result['weighted_f1'],
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), SAVE_DIR / 'best_model.pth')
            logging.info(f"  → New best saved (Val Macro F1: {best_val_f1:.4f})")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                logging.info(f"Early stopping triggered at epoch {epoch+1} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
                break

    logging.info(f"Training complete. Best Val Macro F1: {best_val_f1:.4f}")

    # --- V1 vs V2 Validation Comparison ---
    logging.info("\n=== V1 vs V2 VALIDATION COMPARISON ===")
    v1_val_note = "V1 best validation F1 was 0.4546 (epoch 13/15). See v1/metadata.json for full context."
    logging.info(v1_val_note)
    logging.info(f"V2 best validation Macro F1: {best_val_f1:.4f}")

    # --- Final Val evaluation with best checkpoint ---
    model.load_state_dict(torch.load(SAVE_DIR / 'best_model.pth', map_location=device))
    best_val_result = evaluate(model, val_loader, device, criterion)

    logging.info("V2 Final Validation Metrics (best checkpoint):")
    logging.info(f"  Accuracy:          {best_val_result['accuracy']:.4f}")
    logging.info(f"  Macro Precision:   {best_val_result['macro_precision']:.4f}")
    logging.info(f"  Macro Recall:      {best_val_result['macro_recall']:.4f}")
    logging.info(f"  Macro F1:          {best_val_result['macro_f1']:.4f}")
    logging.info(f"  Weighted F1:       {best_val_result['weighted_f1']:.4f}")
    logging.info(f"  Per-class F1:      {best_val_result['per_class_f1'].tolist()}")
    logging.info(f"  Referable DR Sensitivity: {best_val_result['referable_dr']['sensitivity']:.4f}")
    logging.info(f"  Referable DR Specificity: {best_val_result['referable_dr']['specificity']:.4f}")

    # --- HELD-OUT TEST EVALUATION (one-shot) ---
    logging.info("\n=== HELD-OUT TEST SET (ONE-SHOT EVALUATION) ===")
    test_result = evaluate(model, test_loader, device, criterion)

    logging.info(f"Test Accuracy:        {test_result['accuracy']:.4f}")
    logging.info(f"Test Macro Precision: {test_result['macro_precision']:.4f}")
    logging.info(f"Test Macro Recall:    {test_result['macro_recall']:.4f}")
    logging.info(f"Test Macro F1:        {test_result['macro_f1']:.4f}")
    logging.info(f"Test Weighted F1:     {test_result['weighted_f1']:.4f}")
    logging.info(f"Confusion Matrix:\n{test_result['confusion_matrix']}")
    logging.info(f"Per-class F1: {test_result['per_class_f1'].tolist()}")
    ref = test_result['referable_dr']
    logging.info(f"Referable DR: TP={ref['TP']}, TN={ref['TN']}, FP={ref['FP']}, FN={ref['FN']}")
    logging.info(f"  Sensitivity: {ref['sensitivity']:.4f} | Specificity: {ref['specificity']:.4f}")
    logging.info(f"  Precision: {ref['precision']:.4f} | F1: {ref['f1']:.4f}")
    if ref['roc_auc'] is not None:
        logging.info(f"  Referable ROC-AUC: {ref['roc_auc']:.4f}")

    # --- Save test_predictions.csv ---
    pred_rows = []
    for i in range(len(test_result['labels'])):
        p = test_result['probs'][i]
        true_g = int(test_result['labels'][i])
        pred_g = int(test_result['preds'][i])
        pred_rows.append({
            'image_id': test_dataset.df.iloc[i][test_dataset.img_col],
            'true_grade': true_g,
            'predicted_grade': pred_g,
            'prob_0': float(p[0]),
            'prob_1': float(p[1]),
            'prob_2': float(p[2]),
            'prob_3': float(p[3]),
            'prob_4': float(p[4]),
            'confidence': float(np.max(p)),
            'referable_true': int(true_g >= 2),
            'referable_pred': int(pred_g >= 2),
        })
    preds_df = pd.DataFrame(pred_rows)
    preds_df.to_csv(SAVE_DIR / 'test_predictions.csv', index=False)
    logging.info(f"Saved test_predictions.csv ({len(pred_rows)} rows)")

    # --- Error analysis ---
    grade1_errors = preds_df[preds_df['true_grade'] == 1]
    grade2_overpred = preds_df[(preds_df['predicted_grade'] == 2) & (preds_df['true_grade'] != 2)]
    high_conf_wrong = preds_df[(preds_df['true_grade'] != preds_df['predicted_grade']) & (preds_df['confidence'] >= 0.80)]
    low_conf_all = preds_df[preds_df['confidence'] < 0.40]

    logging.info(f"\nError Analysis:")
    logging.info(f"  Grade 1 samples in test: {len(grade1_errors)} | Correctly recalled: {(grade1_errors['predicted_grade']==1).sum()}")
    logging.info(f"  Grade 2 over-predictions (pred=2, true!=2): {len(grade2_overpred)}")
    logging.info(f"  High-confidence (>=0.80) incorrect: {len(high_conf_wrong)}")
    logging.info(f"  Low-confidence (<0.40) predictions: {len(low_conf_all)}")

    # --- Save validation history ---
    pd.DataFrame(val_metrics_history).to_csv(SAVE_DIR / 'val_history.csv', index=False)

    # --- Save metadata ---
    metadata_v2 = {
        "model_name": "DR-SUGAR-IDRiD-MobileNetV3-V2",
        "architecture": "mobilenet_v3_small",
        "version": "2.0.0",
        "dataset": "IDRiD B. Disease Grading",
        "dataset_samples": len(train_dataset) + len(val_dataset) + len(test_dataset),
        "classes": NUM_CLASSES,
        "class_mapping": {"0": "No DR", "1": "Mild", "2": "Moderate", "3": "Severe", "4": "Proliferative"},
        "preprocessing_version": "v2-224-imagenet-aug",
        "random_seed": SEED,
        "changes_from_v1": [
            "WeightedRandomSampler on training loader (replacing class-weighted loss)",
            "Plain CrossEntropyLoss (no class weights)",
            "Extended training: up to 30 epochs",
            "ReduceLROnPlateau scheduler (patience=4, factor=0.5, metric=val_macro_f1)",
            "Early stopping (patience=8, metric=val_macro_f1)",
        ],
        "training_config": {
            "max_epochs": MAX_EPOCHS,
            "optimizer": "Adam",
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "imbalance_strategy": "WeightedRandomSampler",
            "loss": "CrossEntropyLoss (no class weights)",
            "scheduler": f"ReduceLROnPlateau(patience={SCHEDULER_PATIENCE}, factor={SCHEDULER_FACTOR}, mode='max')",
            "early_stopping_patience": EARLY_STOP_PATIENCE,
        },
        "best_val_macro_f1": float(best_val_f1),
        "checkpoint_path": str(SAVE_DIR / 'best_model.pth'),
        "evaluation_split": "split_test.csv",
        "val_metrics_best_checkpoint": {
            "accuracy": float(best_val_result['accuracy']),
            "macro_precision": float(best_val_result['macro_precision']),
            "macro_recall": float(best_val_result['macro_recall']),
            "macro_f1": float(best_val_result['macro_f1']),
            "weighted_f1": float(best_val_result['weighted_f1']),
            "per_class_f1": best_val_result['per_class_f1'].tolist(),
            "referable_dr": {k: v for k, v in best_val_result['referable_dr'].items()},
        },
        "test_metrics": {
            "accuracy": float(test_result['accuracy']),
            "macro_precision": float(test_result['macro_precision']),
            "macro_recall": float(test_result['macro_recall']),
            "macro_f1": float(test_result['macro_f1']),
            "weighted_f1": float(test_result['weighted_f1']),
            "confusion_matrix": test_result['confusion_matrix'].tolist(),
            "per_class": {
                "precision": test_result['per_class_precision'].tolist(),
                "recall": test_result['per_class_recall'].tolist(),
                "f1": test_result['per_class_f1'].tolist(),
            },
            "referable_dr": {k: v for k, v in test_result['referable_dr'].items()},
        },
        "training_timestamp": datetime.datetime.now().isoformat(),
        "disclaimer": "Research/prototype model evaluation only. Not clinically validated.",
    }

    with open(SAVE_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata_v2, f, indent=2, default=str)

    logging.info(f"\nAll v2 artifacts saved to {SAVE_DIR}/")
    logging.info("Run export_idrid_model_v2.py to produce models/idrid/v2/model.onnx")


if __name__ == '__main__':
    train_model()
