import numpy as np
import pandas as pd

from src.explainability import (
    get_feature_importance,
    get_shap_summary,
    get_local_explanation,
    create_local_summary
)


class MockShapValues:
    def __init__(self, values):
        self.values = values


def test_get_feature_importance():
    values = MockShapValues(
        np.array([
            [2.0, -1.0],
            [4.0, 3.0]
        ])
    )
    result = get_feature_importance(
        values,
        ["feature_1", "feature_2"]
    )
    assert result.iloc[0]["feature"] == "feature_1"
    assert result.iloc[0]["mean_abs_shap"] == 3.0


def test_get_shap_summary():
    values = MockShapValues(
        np.array([
            [2.0, -1.0],
            [4.0, 3.0]
        ])
    )
    result = get_shap_summary(
        values,
        ["feature_1", "feature_2"]
    )
    assert result["total_features"] == 2
    assert result["top_5_features"][0] == "feature_1"


def test_get_local_explanation():
    values = MockShapValues(
        np.array([[2.0, -3.0]])
    )
    result = get_local_explanation(
        values,
        ["feature_1", "feature_2"]
    )
    assert result.iloc[0]["feature"] == "feature_2"


def test_create_local_summary():
    explanation = pd.DataFrame({
        "feature": ["feature_1", "feature_2"],
        "shap_value": [-3.0, 1.0],
        "abs_shap": [3.0, 1.0]
    })
    result = create_local_summary(explanation)
    assert result["most_influential_feature"] == "feature_1"
    assert result["impact"] == "decreases model output"