
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

from src.data.dataset import create_dataloaders
from src.models.coral_resnet import CoralResNet18, coral_loss, coral_predict
from src.evaluation.metrics import calculate_metrics


# -----------------------------
# Configuration
# -----------------------------

SEED = 42
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

NUM_CLASSES = 4
DATA_ROOT = "/content/dfu_data"

CHECKPOINT_DIR = Path("results/checkpoints")
HISTORY_DIR = Path("results")

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Reproducibility
# -----------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Training
# -----------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = coral_loss(logits, labels, NUM_CLASSES)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = coral_predict(logits)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    average_loss = total_loss / len(loader.dataset)

    metrics = calculate_metrics(all_labels, all_predictions)

    return average_loss, metrics


# -----------------------------
# Validation
# -----------------------------

def validate(model, loader, device):
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = coral_loss(logits, labels, NUM_CLASSES)

            total_loss += loss.item() * images.size(0)

            predictions = coral_predict(logits)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    average_loss = total_loss / len(loader.dataset)

    metrics = calculate_metrics(all_labels, all_predictions)

    return average_loss, metrics


# -----------------------------
# Main
# -----------------------------

def main():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cpu":
        print(
            "WARNING: Training on CPU may be slow. "
            "Consider using Google Colab GPU for full training."
        )

    print("\nLoading datasets...")

    train_loader, valid_loader, _ = create_dataloaders(
        data_root=DATA_ROOT,
        batch_size=BATCH_SIZE
    )

    print("\nLoading pretrained ResNet18...")

    model = CoralResNet18(
        num_classes=NUM_CLASSES,
        pretrained=True
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_qwk = -float("inf")
    history = []

    best_checkpoint = CHECKPOINT_DIR / "best_coral_resnet18.pth"

    print("\nStarting training...")

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"{'=' * 50}")

        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device
        )

        valid_loss, valid_metrics = validate(
            model,
            valid_loader,
            device
        )

        print(f"\nTrain loss: {train_loss:.4f}")
        print(f"Validation loss: {valid_loss:.4f}")

        print(
            f"Train accuracy: {train_metrics['accuracy']:.4f}"
        )
        print(
            f"Validation accuracy: {valid_metrics['accuracy']:.4f}"
        )
        print(
            f"Validation macro F1: {valid_metrics['macro_f1']:.4f}"
        )
        print(
            f"Validation QWK: {valid_metrics['qwk']:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": valid_loss,
            "train_accuracy": train_metrics["accuracy"],
            "validation_accuracy": valid_metrics["accuracy"],
            "validation_macro_f1": valid_metrics["macro_f1"],
            "validation_qwk": valid_metrics["qwk"]
        })
        # Save the latest state after every epoch.
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_qwk": valid_metrics["qwk"],
            "validation_loss": valid_loss,
            "seed": SEED
        }, CHECKPOINT_DIR / "last_checkpoint.pth")
        current_qwk = valid_metrics["qwk"]

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

            print("Best model checkpoint saved!")

        # Save training history after every epoch.
        with open(
            HISTORY_DIR / "training_history.csv",
            "w",
            newline=""
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=history[0].keys()
            )
            writer.writeheader()
            writer.writerows(history)

    print("\nTraining complete.")
    print(f"Best validation QWK: {best_qwk:.4f}")
    print(f"Best checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    main()
