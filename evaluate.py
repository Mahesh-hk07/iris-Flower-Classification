"""
==============================================================================
Multi-Class Flower Image Classifier - Evaluation Script
==============================================================================
This script evaluates the trained PyTorch MobileNetV2 model checkpoint
(models/flower_classifier.pt) against the held-out test dataset (data/test).

Computes:
  - Test Accuracy
  - Precision, Recall, and F1-Score per class
  - Actual Confusion Matrix saved to 'results/confusion_matrix.png'
==============================================================================
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "flower_classifier.pt")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CM_SAVE_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.png")
METADATA_PATH = os.path.join(BASE_DIR, "metadata", "flower_metadata.json")


def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Error: Model file not found at {MODEL_PATH}.")
        print("[!] Please run 'python train.py' first.")
        sys.exit(1)

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    num_classes = checkpoint.get("num_classes", 5)
    class_names = checkpoint.get("class_names", [])
    input_size = checkpoint.get("input_size", 224)
    norm_mean = checkpoint.get("norm_mean", [0.485, 0.456, 0.406])
    norm_std = checkpoint.get("norm_std", [0.229, 0.224, 0.225])

    model = models.mobilenet_v2(weights=None)
    in_features = model.last_channel
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, class_names, input_size, norm_mean, norm_std, checkpoint


def evaluate():
    print("=" * 68)
    print("   EVALUATING FLOWER CLASSIFIER ON HELD-OUT TEST DATASET")
    print("=" * 68)

    if not os.path.exists(TEST_DIR):
        print(f"[!] Error: Test dataset folder not found at {TEST_DIR}.")
        sys.exit(1)

    model, class_names, input_size, norm_mean, norm_std, ckpt = load_model()

    # Load metadata for display
    display_names = []
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        display_names = [metadata.get(c, {}).get("display_name", c.capitalize()) for c in class_names]
    else:
        display_names = [c.capitalize() for c in class_names]

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    print(f"[*] Found {len(test_dataset)} held-out test images across {len(class_names)} classes:")
    print(f"    Classes: {display_names}")

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    test_acc = float(np.mean(y_true == y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    print("\n" + "=" * 68)
    print("                    ACTUAL TEST EVALUATION")
    print("=" * 68)
    print(f"\n  Held-Out Test Accuracy: {test_acc * 100:.2f}% ({np.sum(y_true == y_pred)}/{len(y_true)} correct)")
    print("  Notice: Test accuracy is measured on the project's test dataset")
    print("          and does not guarantee real-world performance.\n")

    print("  Confusion Matrix:")
    header = "   " + "".join([f"{name:>13}" for name in display_names])
    print(header)
    for i, row in enumerate(cm):
        row_str = "".join([f"{val:>13}" for val in row])
        print(f"{display_names[i]:<12} {row_str}")

    print("\n  Detailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=display_names, zero_division=0))

    # Plot Confusion Matrix
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=str(cm[i, j]), va='center', ha='center', size='x-large', weight='bold')

    plt.xlabel('Predicted Class', fontsize=12, labelpad=10)
    plt.ylabel('Ground Truth Class', fontsize=12, labelpad=10)
    plt.title(f'Flower Classification Confusion Matrix (Test Acc: {test_acc * 100:.1f}%)', fontsize=12, pad=15)
    ax.set_xticks(range(len(display_names)))
    ax.set_yticks(range(len(display_names)))
    ax.set_xticklabels(display_names, rotation=25, ha='left')
    ax.set_yticklabels(display_names)
    fig.colorbar(cax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(CM_SAVE_PATH, dpi=150)
    plt.close()

    print(f"[*] Confusion matrix plot saved to: {CM_SAVE_PATH}")
    print("=" * 68)

    return test_acc, cm


main = evaluate

if __name__ == "__main__":
    evaluate()

