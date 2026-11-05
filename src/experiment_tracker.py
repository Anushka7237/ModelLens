import json
from pathlib import Path
from datetime import datetime
import uuid

def save_experiment(
    experiment_name,
    applied_fix,
    baseline_metrics,
    fixed_model_metrics,
    performance_comparison,
    explainability_summary=None,
    diagnostic_summary=None
):
    experiment = {
    "experiment_id": str(uuid.uuid4()),
    "experiment_name": experiment_name,
    "applied_fix": applied_fix,
    "baseline_metrics": baseline_metrics,
    "fixed_model_metrics": fixed_model_metrics,
    "performance_comparison": performance_comparison,
    "explainability_summary": explainability_summary,
    "diagnostic_summary": diagnostic_summary,
    "timestamp": datetime.now().isoformat()
    }
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    file_path = data_dir / "experiments.json"
    if file_path.exists():
        with open(file_path, "r") as file:
            experiments = json.load(file)
    else:
        experiments = []
    experiments.append(experiment)
    with open(file_path, "w") as file:
        json.dump(experiments, file, indent=4, default=float)
    return experiment

def load_experiments():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    file_path = data_dir / "experiments.json"
    if not file_path.exists():
        return []
    with open(file_path, "r") as file:
        experiments = json.load(file)
    return experiments


def get_experiment_summary():
    experiments = load_experiments()
    if not experiments:
        return {
            "total_experiments": 0,
            "experiments": []
        }
    summary = []
    for experiment in experiments:
        summary.append({
            "experiment_name": experiment["experiment_name"],
            "applied_fix": experiment["applied_fix"],
            "baseline_f1": experiment["baseline_metrics"]["f1_score"],
            "fixed_f1": experiment["fixed_model_metrics"]["f1_score"],
            "f1_change": experiment["performance_comparison"]["f1_score"]["change"]
        })
    return {
        "total_experiments": len(experiments),
        "experiments": summary
    }
    
def get_best_experiment(metric="f1_score"):
    experiments = load_experiments()
    valid_experiments = [
        exp for exp in experiments
        if metric in exp["fixed_model_metrics"]
    ]
    if not valid_experiments:
        return None
    best_experiment = max(
        valid_experiments,
        key=lambda exp: (
            exp["fixed_model_metrics"][metric],
            exp.get("timestamp", "")
        )
    )
    return {
        "experiment_id": best_experiment.get("experiment_id"),
        "experiment_name": best_experiment["experiment_name"],
        "applied_fix": best_experiment["applied_fix"],
        "metric": metric,
        "best_score": best_experiment["fixed_model_metrics"][metric]
    }
    
def get_experiment_by_id(experiment_id):
    experiments = load_experiments()
    for experiment in experiments:
        if experiment.get("experiment_id") == experiment_id:
            return experiment
    return None


def migrate_experiment_ids():
    experiments = load_experiments()
    updated = False
    for experiment in experiments:
        if not experiment.get("experiment_id"):
            experiment["experiment_id"] = str(uuid.uuid4())
            updated = True
    if updated:
        data_dir = Path(__file__).resolve().parent.parent / "data"
        file_path = data_dir / "experiments.json"
        with open(file_path, "w") as file:
            json.dump(experiments, file, indent=4, default=float)
    return experiments


def rank_experiments(metric="f1_score"):
    experiments = load_experiments()
    valid_experiments = [
        exp for exp in experiments
        if metric in exp.get("fixed_model_metrics", {})
    ]
    ranked = sorted(
        valid_experiments,
        key=lambda exp: (
            exp["fixed_model_metrics"][metric],
            exp.get("timestamp", "")
        ),
        reverse=True
    )
    results = []

    for rank, exp in enumerate(ranked, start=1):
        baseline = exp.get("baseline_metrics", {}).get(metric)
        fixed = exp["fixed_model_metrics"][metric]

        results.append({
            "rank": rank,
            "experiment_id": exp.get("experiment_id"),
            "experiment_name": exp["experiment_name"],
            "applied_fix": exp["applied_fix"],
            "baseline_score": round(baseline, 4) if baseline is not None else None,
            "fixed_score": round(fixed, 4),
            "improvement": round(fixed - baseline, 4) if baseline is not None else None
        })

    return results


def compare_experiments(experiment_id_1, experiment_id_2):
    exp1 = get_experiment_by_id(experiment_id_1)
    exp2 = get_experiment_by_id(experiment_id_2)

    if exp1 is None or exp2 is None:
        raise ValueError("One or both experiment IDs were not found.")
    metrics = set(exp1["fixed_model_metrics"]) & set(
        exp2["fixed_model_metrics"]
    )
    comparison = {}
    for metric in metrics:
        score1 = exp1["fixed_model_metrics"][metric]
        score2 = exp2["fixed_model_metrics"][metric]
        if isinstance(score1, (int, float)) and isinstance(score2, (int, float)):
            comparison[metric] = {
                "experiment_1": round(float(score1), 4),
                "experiment_2": round(float(score2), 4),
                "difference": round(float(score2 - score1), 4)
            }
    return {
        "experiment_1": exp1["experiment_name"],
        "experiment_2": exp2["experiment_name"],
        "metric_comparison": comparison
    }
    
    
def get_experiment_winner(experiment_id_1, experiment_id_2,
                          metric="f1_score", tolerance=0.0001):
    comparison = compare_experiments(
        experiment_id_1,
        experiment_id_2
    )
    metrics = comparison["metric_comparison"]
    if metric not in metrics:
        raise ValueError(f"Metric '{metric}' is not available.")
    score1 = metrics[metric]["experiment_1"]
    score2 = metrics[metric]["experiment_2"]
    difference = round(score2 - score1, 4)
    if abs(difference) <= tolerance:
        winner = "Tie"
        message = "Both experiments performed equally."
    elif difference > 0:
        winner = "Experiment 2"
        message = "Experiment 2 performed better."
    else:
        winner = "Experiment 1"
        message = "Experiment 1 performed better."
    return {
        "metric": metric,
        "experiment_1": comparison["experiment_1"],
        "experiment_2": comparison["experiment_2"],
        "score_1": score1,
        "score_2": score2,
        "difference": difference,
        "winner": winner,
        "message": message
    }
    
def save_experiment_comparison_report(
    experiment_id_1,
    experiment_id_2,
    metric="f1_score"
):
    comparison = compare_experiments(
        experiment_id_1,
        experiment_id_2
    )
    winner = get_experiment_winner(
        experiment_id_1,
        experiment_id_2,
        metric
    )
    report = {
        "generated_at": datetime.now().isoformat(),
        "experiment_ids": {
            "experiment_1": experiment_id_1,
            "experiment_2": experiment_id_2
        },
        "experiment_names": {
            "experiment_1": comparison["experiment_1"],
            "experiment_2": comparison["experiment_2"]
        },
        "metric_comparison": comparison["metric_comparison"],
        "winner_analysis": winner
    }
    data_dir = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "comparison_reports"
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = data_dir / f"comparison_{timestamp}.json"
    with open(file_path, "w") as file:
        json.dump(report, file, indent=4)
    return file_path