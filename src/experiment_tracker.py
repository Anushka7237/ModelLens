import json
from pathlib import Path
from datetime import datetime


def save_experiment(
    experiment_name,
    applied_fix,
    baseline_metrics,
    fixed_model_metrics,
    performance_comparison
):
    experiment = {
        "experiment_name": experiment_name,
        "applied_fix": applied_fix,
        "baseline_metrics": baseline_metrics,
        "fixed_model_metrics": fixed_model_metrics,
        "performance_comparison": performance_comparison,
        "timestamp": datetime.now().isoformat()
    }
    # Store all experiments in the project's data folder
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