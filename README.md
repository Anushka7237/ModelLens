# ModelLens

ModelLens is a Python-based machine learning diagnostics and explainability toolkit for tabular data. It helps users detect data quality issues, evaluate model behavior, explain feature importance, and generate actionable recommendations for improving model performance.

## What ModelLens does

ModelLens focuses on three connected goals:

- Data quality assessment: identify missing values, duplicates, constant features, near-constant features, low-variance signals, high correlations, and outliers.
- Model diagnostics: detect class imbalance, overfitting, underfitting, train/test distribution drift, and possible target leakage.
- Explainable ML workflow: generate SHAP-based feature importance summaries, local explanations, and recommendation reports for decision support.

## Key features

- Dataset profiling and issue detection across tabular features
- Outlier detection using the IQR method
- Model evaluation with accuracy, precision, recall, F1, ROC-AUC, and PR-AUC
- SHAP-based global and local explainability
- Recommendation engine for corrective actions
- JSON report generation for diagnostics, model summaries, and experiment tracking
- Test suite covering core diagnostics and evaluation behavior

## Project structure

```text
ModelLens/
├── data/
│   ├── sample_dataset.csv
│   ├── diagnostic_report.json
│   ├── explainability_report.json
│   ├── model_report.json
│   ├── recommendation_report.json
│   └── report_overview.json
├── notebooks/
│   └── modelens_v1.ipynb
├── src/
│   ├── diagnostics.py
│   ├── evaluation.py
│   ├── explainability.py
│   ├── experiment_tracker.py
│   ├── recommendations.py
│   └── reporting.py
├── tests/
│   ├── test_diagnostics.py
│   ├── test_evaluation.py
│   ├── test_explainability.py
│   ├── test_recommendations.py
│   └── test_reporting.py
├── requirements.txt
├── .gitignore
├── README.md
└── venv
```

## Tech stack

- Python
- pandas, NumPy, SciPy
- scikit-learn
- SHAP
- Pytest
- Jupyter Notebook

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/Anushka7237/ModelLens.git
cd ModelLens
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Quick usage example

The project is structured as a Python toolkit, so you can import the diagnostic and evaluation utilities directly from the `src` package.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from src.diagnostics import run_diagnostics
from src.evaluation import evaluate_classification_model
from src.explainability import calculate_shap_values, create_explainability_report

# Load a dataset
# df = pd.read_csv("data/sample_dataset.csv")
# target = "target"

# Split into train/test sets
# X = df.drop(columns=[target])
# y = df[target]
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# Train a model
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# Run diagnostics and evaluation
# diagnostics = run_diagnostics(df, target, model, X_train, y_train, X_test, y_test)
# metrics = evaluate_classification_model(model, X_test, y_test)

# Explain model behavior
# shap_values = calculate_shap_values(model, X_train, X_test)
# report = create_explainability_report(shap_values, X.columns.tolist())
```

## Notebook workflow

The repository includes a notebook for interactive exploration:

```text
notebooks/modelens_v1.ipynb
```

This notebook walks through dataset diagnostics, model training, outlier remediation experiments, performance comparisons, and SHAP-driven feature analysis.

## Output and reporting

ModelLens writes structured JSON outputs into the `data/` folder, including:

- diagnostic reports
- explanation summaries
- recommendation reports
- overall model summaries

These outputs make it easier to compare model behavior before and after applying fixes.

## Testing

Run the test suite from the project root:

```bash
python -m pytest
```

## Current status

ModelLens is currently a prototype-style ML diagnostics toolkit. It already includes diagnostics, explanation utilities, recommendation generation, and evaluation helpers, and it is designed to be extended as a more complete model-quality and decision-support platform.

## Project goal

The goal of ModelLens is to make machine learning experiments more transparent, interpretable, and evidence-driven by combining automated diagnostics, corrective recommendations, explainability, and experiment tracking in one workflow.
