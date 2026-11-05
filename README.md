# ModelLens

ModelLens is a machine-learning model diagnostics and explainability platform designed to identify potential issues in tabular datasets and trained models. It analyzes data quality, evaluates model performance, provides evidence-based diagnostic insights, and allows users to experiment with corrective actions and compare model performance before and after applying fixes.

## Key Features

- **Dataset Profiling:** Analyze dataset dimensions, data types, missing values, duplicates, and feature statistics.
- **Data Quality Diagnostics:** Detect missing values, duplicate rows, constant and near-constant features, low-variance features, and highly correlated features.
- **Model Diagnostics:** Identify potential class imbalance, overfitting, underfitting, and train/test distribution shifts.
- **Outlier Detection:** Detect outliers using the IQR method and classify their severity based on the percentage of affected values.
- **Data Correction:** Apply IQR-based outlier capping and verify the changes by running diagnostics again.
- **Model Evaluation:** Evaluate classification models using accuracy, precision, recall, and F1-score.
- **Performance Comparison:** Compare baseline and corrected models to understand the impact of data preprocessing on model performance.
- **Explainability with SHAP:** Analyze global feature importance and understand how individual features contribute to model predictions.
- **Experiment Tracking:** Save and retrieve experiment details, applied fixes, model metrics, performance comparisons, and timestamps using JSON.

## Project Structure

```text
ModelLens/
│
├── data/
│   ├── sample_dataset.csv
│   ├── diagnostic_report.json
│   ├── outlier_experiment.json
│   └── experiments.json
│
├── src/
│   ├── diagnostics.py
│   └── experiment_tracker.py
│
├── notebooks/
│   └── modelens_v1.ipynb
│
├── tests/
│   └── test_diagnostics.py
│
├── requirements.txt
└── README.md
```

## Tech Stack

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy, SciPy
- **Machine Learning:** Scikit-learn
- **Explainable AI:** SHAP
- **Data Visualization:** Matplotlib, SHAP plots
- **Experiment Tracking:** JSON
- **Testing:** Pytest
- **Development Environment:** Jupyter Notebook, VS Code
- **Version Control:** Git and GitHub

## Setup and Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd ModelLens
```

### 2. Create a virtual environment

```powershell
python -m venv venv
```

### 3. Activate the virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the notebook

Open the following notebook in VS Code or Jupyter:

```text
notebooks/modelens_v1.ipynb
```

Run the cells sequentially to explore dataset diagnostics, model training, corrective experiments, performance comparisons, and SHAP explanations.

## Current Experiment: IQR Outlier Capping

ModelLens currently includes an outlier-capping experiment using the Interquartile Range (IQR) method.

- **Applied Fix:** IQR-based outlier capping
- **Baseline Accuracy:** 81%
- **Accuracy After Fix:** 80%
- **Baseline F1-score:** 0.81
- **F1-score After Fix:** 0.798

The experiment demonstrates that removing detected data-quality issues does not necessarily improve predictive performance. ModelLens preserves both diagnostic results and performance comparisons to support evidence-based decisions.

## Running Tests

Execute the following command from the project root:

```bash
python -m pytest
```

## Project Goal

The goal of ModelLens is to make machine-learning experiments more transparent and interpretable by combining automated diagnostics, evidence-based recommendations, explainability, corrective experiments, and performance tracking in one platform.

**Current Status:** Prototype under development. Classification diagnostics, IQR-based outlier capping, experiment tracking, and initial SHAP feature-importance analysis are implemented. Further development will expand the corrective-action workflow and platform capabilities.