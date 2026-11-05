import numpy as np
from sklearn.linear_model import LogisticRegression
from src.evaluation import evaluate_classification_model

def test_evaluate_classification_model():
    X = np.array([
        [1, 2], [2, 3], [3, 4], [4, 5],
        [5, 6], [6, 7], [7, 8], [8, 9]
    ])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    model = LogisticRegression()
    model.fit(X, y)
    metrics = evaluate_classification_model(model, X, y)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "confusion_matrix" in metrics