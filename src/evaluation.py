import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


def evaluate_classification_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        probabilities = model.decision_function(X_test)
    else:
        probabilities = None
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist()
    }
    if probabilities is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, probabilities)
        metrics["pr_auc"] = average_precision_score(
            y_test, probabilities
        )
    return {
        key: (
            np.round(value, 4).item()
            if isinstance(value, (np.floating, float))
            else value
        )
        for key, value in metrics.items()
    }