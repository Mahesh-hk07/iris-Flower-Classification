"""
==============================================================================
Multi-Class Flower Image Classifier - Model Training Script
==============================================================================
This script trains a Transfer Learning model (MobileNetV2) on real botanical
flower photographs across multiple flower classes (Hibiscus, Rose, Sunflower,
Lotus, Iris).

Key Workflow:
1. Dynamically discovers classes from the 'data/train/' directory.
2. Applies standard computer vision preprocessing and data augmentation.
3. Loads a pretrained MobileNetV2 convolutional backbone.
4. Attaches a customized classification head for the discovered classes.
5. Trains the model and tracks validation accuracy across epochs.
6. Saves the best model checkpoint to 'models/flower_classifier.pt'.
==============================================================================
"""

import os
import sys
import time
import copy
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "validation")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "flower_classifier.pt")
METADATA_PATH = os.path.join(BASE_DIR, "metadata", "flower_metadata.json")

# Hyperparameters
BATCH_SIZE = 8
NUM_EPOCHS = 18
LEARNING_RATE = 0.001
IMAGE_SIZE = 224

# ImageNet normalization standard
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def verify_dataset():
    """
    Checks that data/train and data/validation exist and contain class subfolders.
    """
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        print("\n" + "!" * 68)
        print("ERROR: Flower image dataset not found!")
        print("Expected structure: data/train/<class>/ and data/validation/<class>/")
        print("Please run: python download_dataset.py")
        print("!" * 68 + "\n")
        sys.exit(1)

    train_classes = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    val_classes = sorted([d for d in os.listdir(VAL_DIR) if os.path.isdir(os.path.join(VAL_DIR, d))])

    if not train_classes:
        print("[!] Error: No class directories found in data/train/.")
        sys.exit(1)

    train_count = sum(len(files) for _, _, files in os.walk(TRAIN_DIR) if files)
    val_count = sum(len(files) for _, _, files in os.walk(VAL_DIR) if files)

    print(f"[*] Verified dataset classes: {train_classes}")
    print(f"    - Training images   : {train_count}")
    print(f"    - Validation images : {val_count}")

    # Cross-reference with metadata
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        for c in train_classes:
            family = metadata.get(c, {}).get("family", "Unknown Family")
            display = metadata.get(c, {}).get("display_name", c.capitalize())
            print(f"    * {display} -> Family: {family}")

    return train_classes


def get_data_loaders():
    """
    Builds training DataLoader with data augmentation and validation DataLoader.
    """
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD)
        ]),
        "validation": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD)
        ]),
    }

    image_datasets = {
        "train": datasets.ImageFolder(TRAIN_DIR, data_transforms["train"]),
        "validation": datasets.ImageFolder(VAL_DIR, data_transforms["validation"])
    }

    data_loaders = {
        "train": DataLoader(image_datasets["train"], batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        "validation": DataLoader(image_datasets["validation"], batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "validation"]}
    class_names = image_datasets["train"].classes

    return data_loaders, dataset_sizes, class_names


def build_model(num_classes):
    """
    Loads pretrained MobileNetV2 and replaces the final linear classifier.
    """
    print(f"\n[*] Instantiating MobileNetV2 transfer learning model for {num_classes} classes...")
    try:
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
        print("    -> Pretrained ImageNet weights loaded.")
    except Exception as e:
        print(f"    -> Warning loading weights online ({e}). Using initialized MobileNetV2.")
        model = models.mobilenet_v2(weights=None)

    # Freeze lower feature extractor layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Unfreeze the top feature block (layers 16 and 17) for botanical domain adaptation
    for param in model.features[16:].parameters():
        param.requires_grad = True

    # Replace classifier head
    in_features = model.last_channel
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )

    return model


def train_loop(model, data_loaders, dataset_sizes, criterion, optimizer, num_epochs=NUM_EPOCHS, device="cpu"):
    """
    Runs the training and validation loops.
    """
    print("\n" + "=" * 68)
    print(f"  STARTING MODEL TRAINING ({num_epochs} Epochs on {str(device).upper()})")
    print("=" * 68)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 28)

        for phase in ["train", "validation"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == "train":
                print(f"  Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc * 100:.2f}%")
            else:
                print(f"  Val Loss  : {epoch_loss:.4f} | Val Acc  : {epoch_acc * 100:.2f}%")
                if epoch_acc >= best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print("\n" + "=" * 68)
    print(f"[*] Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.1f}s")
    print(f"[*] Best Validation Accuracy: {best_acc * 100:.2f}%")
    print("=" * 68)

    model.load_state_dict(best_model_wts)
    return model, best_acc.item()


def save_checkpoint(model, class_names, accuracy):
    """
    Saves model checkpoint and metadata to models/flower_classifier.pt.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    checkpoint = {
        "architecture": "mobilenet_v2",
        "num_classes": len(class_names),
        "class_names": class_names,
        "state_dict": model.state_dict(),
        "input_size": IMAGE_SIZE,
        "norm_mean": NORM_MEAN,
        "norm_std": NORM_STD,
        "validation_accuracy": accuracy,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    torch.save(checkpoint, MODEL_SAVE_PATH)
    size_mb = os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024)
    print(f"\n[*] Model saved to: {MODEL_SAVE_PATH} ({size_mb:.2f} MB)")


def main():
    print("=" * 68)
    print("   FLOWER IMAGE CLASSIFICATION - MULTI-CLASS TRAINING")
    print("=" * 68)

    class_names = verify_dataset()
    data_loaders, dataset_sizes, class_names = get_data_loaders()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {"params": model.features[16:].parameters(), "lr": 0.0001},
        {"params": model.classifier.parameters(), "lr": LEARNING_RATE}
    ])

    trained_model, best_acc = train_loop(
        model, data_loaders, dataset_sizes, criterion, optimizer, num_epochs=NUM_EPOCHS, device=device
    )

    save_checkpoint(trained_model, class_names, best_acc)


if __name__ == "__main__":
    main()
