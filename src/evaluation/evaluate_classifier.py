
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report
)

from src.data.dataset import create_dataloaders
from src.models.resnet_classifier import ResNet18Classifier


DATA_ROOT = "/content/dfu_data"

CHECKPOINT_PATH = Path(
    "results/classifier_checkpoints/best_resnet18_classifier.pth"
)

OUTPUT_DIR = Path("results/classifier_test_evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Grade 1", "Grade 2", "Grade 3", "Grade 4"]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Load test data
_, _, test_loader = create_dataloaders(
    data_root=DATA_ROOT,
    batch_size=32
)


# Load best checkpoint
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False
)

model = ResNet18Classifier(
    num_classes=4,
    pretrained=False
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Checkpoint epoch:", checkpoint["epoch"])
print("Best validation QWK:", checkpoint["validation_qwk"])


# Predict
all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        logits = model(images)
        predictions = logits.argmax(dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.tolist())


y_true = np.asarray(all_labels)
y_pred = np.asarray(all_predictions)


# Metrics
accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true, y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_true, y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_true, y_pred,
    average="macro",
    zero_division=0
)

qwk = cohen_kappa_score(
    y_true, y_pred,
    weights="quadratic"
)

cm = confusion_matrix(
    y_true, y_pred,
    labels=[0, 1, 2, 3]
)


# Print
print("\n" + "=" * 45)
print("STANDARD RESNET18 - TEST RESULTS")
print("=" * 45)

print("Test images:", len(y_true))
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro precision: {precision:.4f}")
print(f"Macro recall: {recall:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"QWK: {qwk:.4f}")

print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1, 2, 3],
    target_names=CLASS_NAMES,
    zero_division=0
))

print("Confusion matrix:")
print(cm)


# Save results
results = {
    "checkpoint_epoch": int(checkpoint["epoch"]),
    "validation_qwk": float(checkpoint["validation_qwk"]),
    "test_images": int(len(y_true)),
    "accuracy": float(accuracy),
    "macro_precision": float(precision),
    "macro_recall": float(recall),
    "macro_f1": float(macro_f1),
    "qwk": float(qwk),
    "confusion_matrix": cm.tolist()
}

with open(OUTPUT_DIR / "test_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

pd.DataFrame({
    "true_label": y_true,
    "predicted_label": y_pred,
    "true_grade": [CLASS_NAMES[i] for i in y_true],
    "predicted_grade": [CLASS_NAMES[i] for i in y_pred]
}).to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False
)

print("\nEvaluation completed. Results saved to:", OUTPUT_DIR)
