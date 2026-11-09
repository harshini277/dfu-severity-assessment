import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix
)


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "qwk": cohen_kappa_score(
            y_true,
            y_pred,
            weights="quadratic"
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1, 2, 3]
        ).tolist()
    }