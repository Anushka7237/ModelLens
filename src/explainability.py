import pandas as pd
import numpy as np
import shap
import json
from pathlib import Path

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

def create_local_summary(local_explanation):
    top_feature = local_explanation.iloc[0]
    summary = {
        "most_influential_feature": top_feature["feature"],
        "shap_value": round(float(top_feature["shap_value"]), 4),
        "impact": (
            "increases model output"
            if top_feature["shap_value"] > 0
            else "decreases model output"
        )
    }
    return summary

def create_explainability_report(
    shap_values,
    feature_names,
    sample_index=0
):
    importance = get_feature_importance(
        shap_values, feature_names
    )
    global_summary = get_shap_summary(
        shap_values, feature_names
    )
    local_explanation = get_local_explanation(
        shap_values, feature_names, sample_index
    )
    local_summary = create_local_summary(local_explanation)
    return {
        "global_summary": global_summary,
        "feature_importance": importance.to_dict(orient="records"),
        "local_summary": local_summary,
        "local_explanation": local_explanation.to_dict(orient="records")
    }
    
def save_explainability_report(report):
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    file_path = data_dir / "explainability_report.json"
    with open(file_path, "w") as file:
        json.dump(report, file, indent=4, default=float)

    return file_path