# ModelLens 🔍
### ML Model Diagnostics, Explainability & Experiment Tracking

ModelLens is a Python-based machine learning diagnostics tool designed to help users identify potential dataset issues, evaluate model performance, understand model predictions, and experiment with corrective techniques.

It provides a structured workflow for diagnosing machine learning problems and making evidence-based decisions through performance comparisons and experiment tracking.

---

## 🚀 Key Features

### 1. Dataset Diagnostics
Automatically identifies common issues in tabular datasets:

- Missing value detection
- Duplicate record detection
- Outlier detection using the IQR method
- Class imbalance detection
- Constant and near-constant feature detection
- Low-variance feature detection
- High feature correlation detection
- Potential target leakage detection
- Distribution shift detection

### 2. Model Performance Evaluation

Evaluates classification models using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix

Also checks for potential overfitting and underfitting.

### 3. Model Explainability

Uses SHAP (SHapley Additive exPlanations) to understand model behavior.

- Global feature importance
- Feature contribution analysis
- Local prediction explanations
- Explainability reports

### 4. Intelligent Recommendations

Generates rule-based recommendations based on detected issues.

Examples:

| Detected Issue | Suggested Action |
|---|---|
| Missing Values | Apply median or mode imputation |
| Outliers | Investigate IQR-based capping |
| Class Imbalance | Try class weights or SMOTE |
| Overfitting | Consider regularization |
| High Correlation | Investigate redundant features |
| Distribution Shift | Investigate differences between training and test data |

Recommendations are suggestions, not automatic guarantees of model improvement.

### 5. Experiment Tracking

- Save machine learning experiments.
- Track applied corrective techniques.
- Store baseline and updated model metrics.
- Rank experiments by selected evaluation metrics.
- Compare two experiments.
- Identify the better-performing experiment.

### 6. Automated Reporting

Generates structured JSON reports containing:

- Diagnostic summaries
- Detected issues
- Feature importance
- Recommendations
- Severity summaries
- Experiment comparisons

---

## 🏗️ Project Architecture

```text
ModelLens/
│
├── data/
│   ├── sample_dataset.csv
│   ├── diagnostic_report.json
│   ├── model_report.json
│   ├── explainability_report.json
│   ├── recommendation_report.json
│   ├── report_overview.json
│   ├── experiments.json
│   └── comparison_reports/
│
├── src/
│   ├── diagnostics.py
│   ├── evaluation.py
│   ├── explainability.py
│   ├── recommendations.py
│   ├── experiment_tracker.py
│   └── reporting.py
│
├── notebooks/
│   └── modellens_v1.ipynb
│
├── tests/
│   ├── test_diagnostics.py
│   ├── test_evaluation.py
│   ├── test_explainability.py
│   ├── test_recommendations.py
│   └── test_reporting.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Anushka7237/ModelLens.git
cd ModelLens
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Project

Open the Jupyter Notebook:

```bash
jupyter notebook
```

Navigate to:

```text
notebooks/modellens_v1.ipynb
```

Run the notebook cells to execute the ModelLens workflow.

---

## 🧪 Running Tests

ModelLens includes automated tests for its core modules.

Run the complete test suite:

```bash
python -m pytest -v
```

**Current test result:** 13 tests passed.

---

## 📊 Sample Experiment Results

An outlier capping experiment was performed using IQR-based treatment.

| Metric | Baseline | After Treatment |
|---|---:|---:|
| Accuracy | 0.8100 | 0.8000 |
| Precision | 0.8100 | 0.8061 |
| Recall | 0.8100 | 0.7900 |
| F1-score | 0.8100 | 0.7980 |
| ROC-AUC | 0.8853 | 0.8860 |
| PR-AUC | 0.8561 | 0.8588 |

**Observation:** Outlier capping did not improve accuracy or F1-score, although ROC-AUC and PR-AUC increased slightly.

This demonstrates why ModelLens compares actual experimental results rather than assuming every corrective technique improves model performance.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Explainable AI:** SHAP
- **Statistical Analysis:** SciPy
- **Visualization:** Matplotlib
- **Testing:** Pytest
- **Development Environment:** Jupyter Notebook
- **Version Control:** Git & GitHub

---

## 🎯 Project Objectives

- Simplify the process of diagnosing machine learning datasets.
- Identify potential problems affecting model performance.
- Improve model interpretability through explainable AI.
- Recommend possible corrective actions.
- Compare experiments using measurable evaluation metrics.
- Maintain a structured record of machine learning experiments.

---
