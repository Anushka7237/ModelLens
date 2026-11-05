import json
from pathlib import Path
from typing import Any

RECOMMENDATION_RULES = {
    "missing_values": {
        "issue": "Missing values detected",
        "action": "Apply missing-value imputation",
        "methods": [
            "Median imputation for numerical features",
            "Mode imputation for categorical features",
            "Consider removing columns with excessive missing values"
        ],
        "priority": "High",
        "reason": "Missing data may reduce model quality or cause training errors."
    },
    "duplicates": {
        "issue": "Duplicate rows detected",
        "action": "Review and remove duplicate records",
        "methods": ["Use df.drop_duplicates() after verifying duplicates"],
        "priority": "Medium",
        "reason": "Duplicate records can bias model training and evaluation."
    },
    "outliers": {
        "issue": "Potential outliers detected",
        "action": "Investigate outliers and test robust handling",
        "methods": [
            "IQR-based capping",
            "Log transformation where appropriate",
            "Robust scaling",
            "Keep legitimate extreme values"
        ],
        "priority": "Medium",
        "reason": "Extreme values may affect some models, but they can also represent valid data."
    },
    "class_imbalance": {
        "issue": "Class imbalance detected",
        "action": "Test imbalance-handling techniques",
        "methods": [
            "Class weights",
            "SMOTE on training data only",
            "Threshold tuning",
            "Evaluate using F1, recall, and PR-AUC"
        ],
        "priority": "High",
        "reason": "The model may perform poorly on minority-class samples."
    },
    "overfitting": {
        "issue": "Potential overfitting detected",
        "action": "Improve generalization",
        "methods": [
            "Apply regularization",
            "Reduce model complexity",
            "Use cross-validation",
            "Collect more training data"
        ],
        "priority": "High",
        "reason": "The model performs better on training data than unseen data."
    },
    "underfitting": {
        "issue": "Potential underfitting detected",
        "action": "Increase model learning capacity",
        "methods": [
            "Try a more expressive model",
            "Improve feature engineering",
            "Reduce excessive regularization"
        ],
        "priority": "Medium",
        "reason": "The model may be failing to capture important patterns."
    },
    "high_correlations": {
        "issue": "Highly correlated features detected",
        "action": "Review correlated features",
        "methods": [
            "Remove redundant features where appropriate",
            "Use regularization",
            "Consider dimensionality reduction"
        ],
        "priority": "Medium",
        "reason": "Redundant features may affect interpretability or some model types."
    },
    "distribution_shift": {
        "issue": "Possible train-test distribution shift",
        "action": "Investigate differences between training and test data",
        "methods": [
            "Check data splitting strategy",
            "Inspect temporal or group-based patterns",
            "Verify preprocessing consistency"
        ],
        "priority": "High",
        "reason": "Training and test data may represent different distributions."
    },
    "constant_features": {
        "issue": "Constant features detected",
        "action": "Remove features with only one unique value",
        "methods": [
            "Check whether the feature contains a single value",
            "Remove constant columns after verification"
        ],
        "priority": "Medium",
        "reason": "Constant features provide no useful information to the model."
    },
    "near_constant_features": {
        "issue": "Near-constant features detected",
        "action": "Review features with very low variation",
        "methods": [
            "Check feature value distributions",
            "Remove features only if they provide no meaningful signal"
        ],
        "priority": "Low",
        "reason": "Features with very little variation may contribute limited predictive value."
    },

    "low_variance_features": {
        "issue": "Low-variance features detected",
        "action": "Review low-variance features",
        "methods": [
            "Analyze feature variance",
            "Consider removing uninformative features",
            "Validate changes using model performance"
        ],
        "priority": "Low",
        "reason": "Low-variance features may provide limited useful information."
    },

    "target_leakage": {
        "issue": "Potential target leakage detected",
        "action": "Investigate suspicious features",
        "methods": [
            "Check whether features contain information unavailable at prediction time",
            "Review feature-target relationships",
            "Remove leakage-causing features if confirmed"
        ],
        "priority": "High",
        "reason": "Target leakage can produce misleadingly high evaluation scores."
    }
}


def generate_recommendations(diagnostic_results: dict[str, Any]):
    """
    Generate corrective-action recommendations from diagnostic flags.
    Accepts either a diagnostic status dictionary or a report
    containing a nested 'status' dictionary.
    """
    if "status" in diagnostic_results:
        diagnostic_results = diagnostic_results["status"]
    recommendations = []
    for diagnostic, detected in diagnostic_results.items():
        if not detected:
            continue
        rule = RECOMMENDATION_RULES.get(diagnostic)
        if rule is None:
            continue
        recommendations.append({
            "diagnostic": diagnostic,
            "issue": rule["issue"],
            "reason": rule["reason"],
            "recommended_action": rule["action"],
            "suggested_methods": rule["methods"],
            "priority": rule["priority"],
            "requires_user_review": True
        })
    priority_order = {
        "High": 1,
        "Medium": 2,
        "Low": 3
    }
    recommendations.sort(
        key=lambda item: priority_order.get(item["priority"], 4)
    )
    return recommendations


def create_recommendation_summary(recommendations):
    return {
        "total_recommendations": len(recommendations),
        "high_priority": sum(
            item["priority"] == "High"
            for item in recommendations
        ),
        "medium_priority": sum(
            item["priority"] == "Medium"
            for item in recommendations
        ),
        "recommendations": recommendations
    }



def save_recommendation_report(summary):
    data_dir = (
        Path(__file__).resolve().parent.parent / "data"
    )
    data_dir.mkdir(exist_ok=True)
    file_path = data_dir / "recommendation_report.json"
    with open(file_path, "w") as file:
        json.dump(summary, file, indent=4)
    return file_path

