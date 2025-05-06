import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    classification_report,
    f1_score,
)
from src.config.config import LABELS

def calculate_metrics(y_true, y_pred_proba, threshold=0.5):
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)

    if y_true.ndim != 2 or y_pred_proba.ndim != 2:
        raise ValueError("y_true and y_pred_proba must be 2D arrays")
    if y_true.shape != y_pred_proba.shape:
        raise ValueError("y_true and y_pred_proba must have the same shape")

    y_pred_binary = (y_pred_proba >= threshold).astype(int)

    report_dict = {}
    full_report = None

    try:
        subset_accuracy = accuracy_score(y_true, y_pred_binary)
        hamming = hamming_loss(y_true, y_pred_binary)
        f1_micro = f1_score(y_true, y_pred_binary, average='micro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred_binary, average='weighted', zero_division=0)
        f1_samples = f1_score(y_true, y_pred_binary, average='samples', zero_division=0)

        report_dict = {
            "subset_accuracy": subset_accuracy,
            "hamming_loss": hamming,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "f1_samples": f1_samples,
        }

        full_report = classification_report(
            y_true,
            y_pred_binary,
            target_names=LABELS,
            output_dict=True,
            zero_division=0
        )

        report_dict["classification_report"] = full_report

    except Exception as e:
        print(f"Error calculating metrics: {e}")
        report_dict["error"] = str(e)

    return report_dict

def classification_report_to_dataframe(report_dict):
    if "classification_report" not in report_dict:
        return pd.DataFrame()

    report_data = report_dict["classification_report"]
    labels = [l for l in report_data.keys() if l in LABELS]
    data = {label: report_data[label] for label in labels}

    df = pd.DataFrame.from_dict(data, orient='index')
    return df