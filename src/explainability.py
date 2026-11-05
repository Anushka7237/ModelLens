import pandas as pd
import numpy as np
import shap


def calculate_shap_values(model, X_train, X_test):
    explainer = shap.LinearExplainer(
        model,
        X_train
    )
    shap_values = explainer(X_test)
    return shap_values


def get_feature_importance(shap_values, feature_names):
    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values.values).mean(axis=0)
    })
    importance = importance.sort_values(
        by="mean_abs_shap",
        ascending=False
    ).reset_index(drop=True)
    return importance

def get_shap_summary(shap_values, feature_names):
    summary = {
        "total_features": len(feature_names),
        "top_5_features": (
            get_feature_importance(shap_values, feature_names)
            .head(5)["feature"]
            .tolist()
        )
    }
    return summary

def get_local_explanation(shap_values, feature_names, sample_index=0):
    values = shap_values.values[sample_index]
    explanation = pd.DataFrame({
        "feature": feature_names,
        "shap_value": values,
        "abs_shap": np.abs(values)
    })
    explanation = explanation.sort_values(
        by="abs_shap",
        ascending=False
    ).reset_index(drop=True)
    return explanation