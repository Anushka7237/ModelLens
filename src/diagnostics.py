import pandas as pd

def check_missing_values(df):
    missing = df.isnull().sum()

    return {
        column: int(count)
        for column, count in missing.items()
        if count > 0
    }

def get_missing_value_evidence(df):
    missing_counts = check_missing_values(df)
    total_rows = len(df)
    evidence = {}
    for feature, count in missing_counts.items():
        percentage = (count / total_rows) * 100
        evidence[feature] = {
            "count": count,
            "percentage": round(percentage, 2)
        }
    return evidence


def add_missing_value_severity(missing_evidence):
    for feature, data in missing_evidence.items():
        percentage = data["percentage"]
        if percentage > 5:
            severity = "high"
        elif percentage >= 1:
            severity = "medium"
        else:
            severity = "low"
        data["severity"] = severity
        
        if severity == "high":
            data["reason"] = "More than 5% of the values are missing."
        elif severity == "medium":
            data["reason"] = "Between 1% and 5% of the values are missing."
        else:
            data["reason"] = "Less than 1% of the values are missing."
    return missing_evidence


def check_duplicates(df):
    return int(df.duplicated().sum())

def check_constant_features(df, target):
    features = df.drop(columns=[target])

    return [
        column
        for column in features.columns
        if features[column].nunique() == 1
    ]


# It helps identify features that have very little variation and may not contribute much to the model.
def check_near_constant_features(df, target, threshold=0.95):
    features = df.drop(columns=[target])

    near_constant = []
    for column in features.columns:
        most_common_ratio = features[column].value_counts(normalize=True).iloc[0]

        if most_common_ratio >= threshold:
            near_constant.append(column)

    return near_constant

# Strongly correlated features can indicate redundant information in the dataset. ModelLens should identify these pairs as a potential feature-quality issue.
def check_high_correlations(df, target, threshold=0.8):
    features = df.drop(columns=[target])
    correlation = features.corr()

    high_correlations = []
    for i in range(len(correlation.columns)):
        for j in range(i + 1, len(correlation.columns)):
            value = correlation.iloc[i, j]

            if abs(value) >= threshold:
                high_correlations.append({
                    "feature_1": correlation.columns[i],
                    "feature_2": correlation.columns[j],
                    "correlation": round(value, 2)
                })

    return high_correlations

def check_outliers(df, target):
    features = df.drop(columns=[target])
    outlier_counts = {}
    for column in features.columns:
        Q1 = features[column].quantile(0.25)
        Q3 = features[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        count = ((features[column] < lower) | (features[column] > upper)).sum()

        if count > 0:
            outlier_counts[column] = int(count)

    return outlier_counts


def get_outlier_evidence(df, target):
    outlier_counts = check_outliers(df, target)
    total_rows = len(df)
    evidence = {}
    for feature, count in outlier_counts.items():
        percentage = (count / total_rows) * 100
        evidence[feature] = {
            "count": count,
            "percentage": round(percentage, 2),
            "non_outlier_count": total_rows - count
        }
    return evidence

# diagnostic_results contains the actual evidence/counts, while diagnostic_status gives a simple True/False summary of whether each issue exists.
def run_diagnostics(df, target, model, X_train, y_train, X_test, y_test):
    return {
        "missing_values": add_missing_value_severity(get_missing_value_evidence(df)),
        "duplicate_rows": get_duplicate_evidence(df),
        "constant_features": check_constant_features(df, target),
        "near_constant_features": check_near_constant_features(df, target),
        "high_correlations": check_high_correlations(df, target),
        "outliers": add_outlier_severity(get_outlier_evidence(df, target)),
        "class_imbalance": {**check_class_imbalance(df, target),"evidence": get_class_imbalance_evidence(df, target)},
        "overfitting": check_overfitting(
            model, X_train, y_train, X_test, y_test
        ),
        "underfitting": check_underfitting(
            model, X_train, y_train
        ),
        "distribution_shift": check_distribution_shift(
            X_train, X_test
        ),
        "low_variance_features": check_low_variance_features(
            df, target
        ),
        "target_leakage": check_target_leakage(
            df, target
        ),
    }

def get_diagnostic_status(diagnostic_results):
    status = {}

    status["missing_values"] = bool(diagnostic_results["missing_values"])
    status["duplicate_rows"] = diagnostic_results["duplicate_rows"]["detected"]
    status["constant_features"] = bool(diagnostic_results["constant_features"])
    status["near_constant_features"] = bool(diagnostic_results["near_constant_features"])
    status["high_correlations"] = bool(diagnostic_results["high_correlations"])
    status["outliers"] = bool(diagnostic_results["outliers"])
    status["class_imbalance"] = bool(diagnostic_results["class_imbalance"]["detected"])
    status["overfitting"] = bool(diagnostic_results["overfitting"]["detected"])
    status["underfitting"] = bool(diagnostic_results["underfitting"]["detected"])
    status["distribution_shift"] = bool(diagnostic_results["distribution_shift"])
    status["low_variance_features"] = bool(diagnostic_results["low_variance_features"])
    status["target_leakage"] = bool(diagnostic_results["target_leakage"])
    return status

def check_class_imbalance(df, target, threshold=0.20):
    class_distribution = df[target].value_counts(normalize=True)
    minority_ratio = class_distribution.min()
    minority_class = class_distribution.idxmin()
    return {
        "detected": minority_ratio < threshold,
        "minority_class": minority_class,
        "minority_ratio": round(float(minority_ratio), 4),
        "distribution": class_distribution.to_dict()
    }
    
def get_class_imbalance_evidence(df, target):
    class_distribution = df[target].value_counts(normalize=True)
    minority_class = class_distribution.idxmin()
    majority_class = class_distribution.idxmax()
    return {
        "minority_class": minority_class,
        "minority_percentage": round(
            float(class_distribution.min() * 100), 2
        ),
        "majority_class": majority_class,
        "majority_percentage": round(
            float(class_distribution.max() * 100), 2
        ),
        "distribution": {
            str(class_name): round(float(percentage * 100), 2)
            for class_name, percentage in class_distribution.items()
        }
    }

# overfitting/underfitting based on training vs validation/test performance
from sklearn.metrics import accuracy_score
def check_overfitting(model, X_train, y_train, X_test, y_test, threshold=0.10):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    gap = train_accuracy - test_accuracy

    return {
        "detected": gap >= threshold,
        "train_accuracy": round(float(train_accuracy), 4),
        "test_accuracy": round(float(test_accuracy), 4),
        "gap": round(float(gap), 4)
    }

def check_underfitting(model, X_train, y_train, threshold=0.70):
    train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    return {
        "detected": train_accuracy < threshold,
        "train_accuracy": round(float(train_accuracy), 4)
    }


# If the distributions differ significantly, ModelLens records that feature as a potential distribution shift.checking whether a feature's distribution is substantially different between the training and test sets
from scipy.stats import ks_2samp
def check_distribution_shift(X_train, X_test, threshold=0.05):
    shift_results = {}
    for column in X_train.columns:
        if X_train[column].dtype in ["int64", "float64"]:
            statistic, p_value = ks_2samp(
                X_train[column],
                X_test[column]
            )
            if p_value < threshold:
                shift_results[column] = {
                    "statistic": round(float(statistic), 4),
                    "p_value": round(float(p_value), 4)
                }

    return shift_results


# A feature with extremely little variation may provide limited information to the model. This is part of the feature-quality diagnostics in the V1 design.
def check_low_variance_features(df, target, threshold=0.01):
    features = df.drop(columns=[target])
    low_variance = []
    for column in features.columns:
        if features[column].dtype in ["int64", "float64"]:
            variance = features[column].var()
            if variance < threshold:
                low_variance.append({
                    "feature": column,
                    "variance": round(float(variance), 6)
                })

    return low_variance

# This is an important part of ModelLens because your specification says the system should identify suspicious leakage situations, not automatically claim that leakage exists.
def check_target_leakage(df, target, threshold=0.95):
    features = df.drop(columns=[target])
    leakage_warnings = []
    if not pd.api.types.is_numeric_dtype(df[target]):
        return leakage_warnings
    for column in features.columns:
        if pd.api.types.is_numeric_dtype(features[column]):
            correlation = features[column].corr(df[target])
            if abs(correlation) >= threshold:
                leakage_warnings.append({
                    "feature": column,
                    "target_correlation": round(float(correlation), 4)
                })

    return leakage_warnings



def add_outlier_severity(outlier_evidence):
    HIGH_THRESHOLD = 5
    MEDIUM_THRESHOLD = 1
    for feature, data in outlier_evidence.items():
        percentage = data["percentage"]
        if percentage >= HIGH_THRESHOLD:
            severity = "high"
        elif percentage >= MEDIUM_THRESHOLD:
            severity = "medium"
        else:
            severity = "low"
            
        data["severity"] = severity
        if severity == "high":
            data["reason"] = "More than 5% of the values are detected as outliers."
        elif severity == "medium":
            data["reason"] = "Between 1% and 5% of the values are detected as outliers."
        else:
            data["reason"] = "Less than 1% of the values are detected as outliers."
    return outlier_evidence


def get_duplicate_evidence(df):
    duplicate_count = check_duplicates(df)
    total_rows = len(df)
    return {
        "count": duplicate_count,
        "percentage": round(
            (duplicate_count / total_rows) * 100, 2
        ),
        "detected": duplicate_count > 0
    }
    
    
def cap_outliers(df, target):
    df_fixed = df.copy()
    features = df_fixed.drop(columns=[target])
    for column in features.columns:
        if not pd.api.types.is_numeric_dtype(features[column]):
            continue
        Q1 = features[column].quantile(0.25)
        Q3 = features[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_fixed[column] = df_fixed[column].clip(
            lower=lower,
            upper=upper
        )
    return df_fixed


def create_diagnostic_summary(diagnostic_results):
    summary = []
    if diagnostic_results["missing_values"]:
        highest_severity = "low"
        for data in diagnostic_results["missing_values"].values():
            if data["severity"] == "high":
                highest_severity = "high"
                break
            elif data["severity"] == "medium":
                highest_severity = "medium"
        summary.append({
            "issue": "Missing Values",
            "status": "detected",
            "severity": highest_severity,
            "evidence": diagnostic_results["missing_values"]
        })
    if diagnostic_results["duplicate_rows"]["detected"]:
        summary.append({
            "issue": "Duplicate Rows",
            "status": "detected",
            "severity": "low",
            "evidence": diagnostic_results["duplicate_rows"]
        })
    if diagnostic_results["outliers"]:
        highest_severity = "low"
        for data in diagnostic_results["outliers"].values():
            if data["severity"] == "high":
                highest_severity = "high"
                break
            elif data["severity"] == "medium":
                highest_severity = "medium"
        summary.append({
            "issue": "Outliers",
            "status": "detected",
            "severity": highest_severity,
            "evidence": diagnostic_results["outliers"]
        })
    if diagnostic_results["class_imbalance"]["detected"]:
        summary.append({
            "issue": "Class Imbalance",
            "status": "detected",
            "severity": "high",
            "evidence": diagnostic_results["class_imbalance"]["evidence"]
        })
    if diagnostic_results["overfitting"]["detected"]:
        summary.append({
            "issue": "Overfitting",
            "status": "detected",
            "severity": "high",
            "evidence": diagnostic_results["overfitting"]
        })
    if diagnostic_results["underfitting"]["detected"]:
        summary.append({
            "issue": "Underfitting",
            "status": "detected",
            "severity": "medium",
            "evidence": diagnostic_results["underfitting"]
        })
    if diagnostic_results["distribution_shift"]:
        summary.append({
            "issue": "Distribution Shift",
            "status": "detected",
            "severity": "medium",
            "evidence": diagnostic_results["distribution_shift"]
        })
    if diagnostic_results["target_leakage"]:
        summary.append({
            "issue": "Potential Target Leakage",
            "status": "warning",
            "severity": "high",
            "evidence": diagnostic_results["target_leakage"]
        })
    if diagnostic_results["constant_features"]:
        summary.append({
            "issue": "Constant Features",
            "status": "detected",
            "severity": "medium",
            "evidence": diagnostic_results["constant_features"]
        })
    if diagnostic_results["near_constant_features"]:
        summary.append({
            "issue": "Near-Constant Features",
            "status": "detected",
            "severity": "low",
            "evidence": diagnostic_results["near_constant_features"]
        })
    if diagnostic_results["high_correlations"]:
        summary.append({
            "issue": "High Feature Correlation",
            "status": "detected",
            "severity": "medium",
            "evidence": diagnostic_results["high_correlations"]
        })
    if diagnostic_results["low_variance_features"]:
        summary.append({
            "issue": "Low Variance Features",
            "status": "detected",
            "severity": "low",
            "evidence": diagnostic_results["low_variance_features"]
        })
    return summary


def create_diagnostic_report(diagnostic_results):
    status = get_diagnostic_status(diagnostic_results)
    summary = create_diagnostic_summary(diagnostic_results)
    return {
        "status": status,
        "summary": summary
    }
    
def compare_model_performance(baseline_metrics, fixed_metrics):
    comparison = {}
    for metric in baseline_metrics:
        before = baseline_metrics[metric]
        after = fixed_metrics[metric]
        # Compare only numeric metrics
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            comparison[metric] = {
                "before": round(float(before), 4),
                "after": round(float(after), 4),
                "change": round(float(after - before), 4)
            }
    return comparison