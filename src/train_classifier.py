
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from src.data.dataset import create_dataloaders
from src.models.resnet_classifier import ResNet18Classifier
from src.evaluation.metrics import calculate_metrics


SEED = 42
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

NUM_CLASSES = 4
DATA_ROOT = "/content/dfu_data"

CHECKPOINT_DIR = Path("results/classifier_checkpoints")
HISTORY_DIR = Path("results")

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for images, labels in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = logits.argmax(dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    average_loss = total_loss / len(loader.dataset)
    metrics = calculate_metrics(all_labels, all_predictions)

    return average_loss, metrics


def main():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    train_loader, valid_loader, _ = create_dataloaders(
        data_root=DATA_ROOT,
        batch_size=BATCH_SIZE
    )

    model = ResNet18Classifier(
        num_classes=NUM_CLASSES,
        pretrained=True
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_qwk = -float("inf")
    history = []

    best_checkpoint = (
        CHECKPOINT_DIR / "best_resnet18_classifier.pth"
    )

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss, train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer
        )

        valid_loss, valid_metrics = run_epoch(
            model, valid_loader, criterion, device
        )

        current_qwk = valid_metrics["qwk"]

        print(f"Train loss: {train_loss:.4f}")
        print(f"Validation loss: {valid_loss:.4f}")
        print(f"Train accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Validation accuracy: {valid_metrics['accuracy']:.4f}")
        print(f"Validation macro F1: {valid_metrics['macro_f1']:.4f}")
        print(f"Validation QWK: {current_qwk:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": valid_loss,
            "train_accuracy": train_metrics["accuracy"],
            "validation_accuracy": valid_metrics["accuracy"],
            "validation_macro_f1": valid_metrics["macro_f1"],
            "validation_qwk": current_qwk
        })

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_qwk": current_qwk,
            "validation_loss": valid_loss,
            "seed": SEED
        }, CHECKPOINT_DIR / "last_classifier_checkpoint.pth")

        if np.isfinite(current_qwk) and current_qwk > best_qwk:
            best_qwk = current_qwk

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_qwk": best_qwk,
                "validation_loss": valid_loss,
                "seed": SEED
            }, best_checkpoint)

            print("Best classifier checkpoint saved!")

        with open(
            HISTORY_DIR / "classifier_training_history.csv",
            "w",
            newline=""
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=history[0].keys()
            )
            writer.writeheader()
            writer.writerows(history)

    print("\nClassifier training complete.")
    print("Best validation QWK:", best_qwk)


if __name__ == "__main__":
    main()
