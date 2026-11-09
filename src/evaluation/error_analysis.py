
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix


# --------------------------------------------------
# Paths
# --------------------------------------------------

CORAL_PATH = Path(
    "results/test_evaluation/test_predictions.csv"
)

STANDARD_PATH = Path(
    "results/classifier_test_evaluation/test_predictions.csv"
)

OUTPUT_DIR = Path("results/error_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "Grade 1",
    "Grade 2",
    "Grade 3",
    "Grade 4"
]


# --------------------------------------------------
# Load predictions
# --------------------------------------------------

coral = pd.read_csv(CORAL_PATH)
standard = pd.read_csv(STANDARD_PATH)

# Confirm both files refer to the same test examples
if not np.array_equal(
    coral["true_label"].to_numpy(),
    standard["true_label"].to_numpy()
):
    raise ValueError("The test labels do not match between models.")

if len(coral) != len(standard):
    raise ValueError("The prediction files have different lengths.")

df = pd.DataFrame({
    "true_label": coral["true_label"],
    "coral_prediction": coral["predicted_label"],
    "standard_prediction": standard["predicted_label"]
})

df["true_grade"] = df["true_label"].map(
    dict(enumerate(CLASS_NAMES))
)

df["coral_grade"] = df["coral_prediction"].map(
    dict(enumerate(CLASS_NAMES))
)

df["standard_grade"] = df["standard_prediction"].map(
    dict(enumerate(CLASS_NAMES))
)


# --------------------------------------------------
# Correct / incorrect predictions
# --------------------------------------------------

df["coral_correct"] = (
    df["true_label"] == df["coral_prediction"]
)

df["standard_correct"] = (
    df["true_label"] == df["standard_prediction"]
)

both_correct = (
    df["coral_correct"] & df["standard_correct"]
).sum()

both_wrong = (
    ~df["coral_correct"] & ~df["standard_correct"]
).sum()

coral_only_correct = (
    df["coral_correct"] & ~df["standard_correct"]
).sum()

standard_only_correct = (
    ~df["coral_correct"] & df["standard_correct"]
).sum()


print("\n" + "=" * 55)
print("MODEL AGREEMENT AND ERROR ANALYSIS")
print("=" * 55)

print("Total test images:", len(df))
print("Both models correct:", both_correct)
print("Both models wrong:", both_wrong)
print("Only CORAL correct:", coral_only_correct)
print("Only standard classifier correct:", standard_only_correct)


# --------------------------------------------------
# Grade-wise error analysis
# --------------------------------------------------

grade_summary = df.groupby("true_grade").agg(
    total_images=("true_label", "size"),
    coral_correct=("coral_correct", "sum"),
    standard_correct=("standard_correct", "sum")
)

grade_summary["coral_accuracy"] = (
    grade_summary["coral_correct"]
    / grade_summary["total_images"]
)

grade_summary["standard_accuracy"] = (
    grade_summary["standard_correct"]
    / grade_summary["total_images"]
)

print("\nGRADE-WISE ACCURACY")
print(grade_summary.round(4))

grade_summary.to_csv(
    OUTPUT_DIR / "grade_wise_analysis.csv"
)


# --------------------------------------------------
# Ordinal error distance
# --------------------------------------------------

df["coral_error_distance"] = abs(
    df["true_label"] - df["coral_prediction"]
)

df["standard_error_distance"] = abs(
    df["true_label"] - df["standard_prediction"]
)

print("\nAVERAGE ORDINAL ERROR DISTANCE")

print(
    "CORAL:",
    round(df["coral_error_distance"].mean(), 4)
)

print(
    "Standard:",
    round(df["standard_error_distance"].mean(), 4)
)


# --------------------------------------------------
# Error categories
# --------------------------------------------------

def categorize_error(row, prediction_column):
    error = abs(
        row["true_label"] - row[prediction_column]
    )

    if error == 0:
        return "Correct"

    elif error == 1:
        return "Adjacent-grade error"

    else:
        return "Large grade error"


df["coral_error_type"] = df.apply(
    lambda row: categorize_error(row, "coral_prediction"),
    axis=1
)

df["standard_error_type"] = df.apply(
    lambda row: categorize_error(row, "standard_prediction"),
    axis=1
)

print("\nCORAL ERROR DISTRIBUTION")
print(df["coral_error_type"].value_counts())

print("\nSTANDARD ERROR DISTRIBUTION")
print(df["standard_error_type"].value_counts())


# --------------------------------------------------
# Confusion matrices
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

coral_cm = confusion_matrix(
    df["true_label"],
    df["coral_prediction"],
    labels=[0, 1, 2, 3]
)

standard_cm = confusion_matrix(
    df["true_label"],
    df["standard_prediction"],
    labels=[0, 1, 2, 3]
)

sns.heatmap(
    coral_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    ax=axes[0]
)

axes[0].set_title("ResNet18 + CORAL")
axes[0].set_xlabel("Predicted grade")
axes[0].set_ylabel("Actual grade")

sns.heatmap(
    standard_cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    ax=axes[1]
)

axes[1].set_title("Standard ResNet18")
axes[1].set_xlabel("Predicted grade")
axes[1].set_ylabel("Actual grade")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# --------------------------------------------------
# Save detailed predictions
# --------------------------------------------------

df.to_csv(
    OUTPUT_DIR / "combined_error_analysis.csv",
    index=False
)

print("\nError analysis saved to:", OUTPUT_DIR.resolve())
print("ANALYSIS COMPLETE!")
