import json
from pathlib import Path
from src.diagnostics import create_diagnostic_report
from src.diagnostics import get_diagnostic_status


def create_model_report(
    diagnostic_results,
    explainability_report,
    recommendation_summary=None
):
    report = {
        "diagnostics": diagnostic_results,
        "explainability": explainability_report,
        "recommendations": recommendation_summary or {
            "total_recommendations": 0,
            "high_priority": 0,
            "medium_priority": 0,
            "recommendations": []
        }
    }
    return report

def save_model_report(report):
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    file_path = data_dir / "model_report.json"
    with open(file_path, "w") as file:
        json.dump(report, file, indent=4, default=float)
    return file_path

def load_model_report():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    file_path = data_dir / "model_report.json"
    if not file_path.exists():
        raise FileNotFoundError("Model report not found.")
    with open(file_path, "r") as file:
        return json.load(file)
    


def create_report_overview(report):
    diagnostics = report.get("diagnostics", {})
    explainability = report.get("explainability", {})
    recommendation_data = report.get("recommendations", {})
    status = diagnostics.get("status")
    if status is None:
        status = get_diagnostic_status(diagnostics)
    detected_issues = [
        issue for issue, detected in status.items()
        if detected
    ]
    global_summary = explainability.get("global_summary", {})
    severity_summary = create_severity_summary({
        "detected_issues": detected_issues
    })

    return {
        "total_issues_detected": len(detected_issues),
        "detected_issues": detected_issues,

        "total_features": global_summary.get("total_features", 0),
        "top_features": global_summary.get("top_5_features", []),
        "explainability_available": bool(explainability),

        "total_recommendations": recommendation_data.get(
            "total_recommendations", 0
        ),
        "high_priority": recommendation_data.get("high_priority", 0),
        "medium_priority": recommendation_data.get("medium_priority", 0),
        "top_recommendations": [
            {
                "issue": item["issue"],
                "priority": item["priority"],
                "action": item["recommended_action"]
            }
            for item in recommendation_data.get("recommendations", [])[:3]
        ],
        "severity_summary": severity_summary
    }




def save_report_overview(overview):
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    file_path = data_dir / "report_overview.json"
    with open(file_path, "w") as file:
        json.dump(overview, file, indent=4)
    return file_path


def create_severity_summary(overview):
    detected_issues = overview.get("detected_issues", [])
    severity_mapping = {
        "missing_values": "High",
        "class_imbalance": "High",
        "target_leakage": "High",
        "overfitting": "High",
        "distribution_shift": "High",
        "duplicates": "Medium",
        "outliers": "Medium",
        "constant_features": "Medium",
        "high_correlations": "Medium",
        "underfitting": "Medium",
        "near_constant_features": "Low",
        "low_variance_features": "Low"
    }
    summary = {
        "High": [],
        "Medium": [],
        "Low": []
    }

    for issue in detected_issues:
        severity = severity_mapping.get(issue, "Low")
        summary[severity].append(issue)

    return {
        "high_severity_count": len(summary["High"]),
        "medium_severity_count": len(summary["Medium"]),
        "low_severity_count": len(summary["Low"]),
        "severity_details": summary
    }
