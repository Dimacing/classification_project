import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random

try:
    from src.config.config import MODEL_DIR, LABELS, DATA_DIR
    from src.data.dataset import load_dataset, prepare_labels
    from src.models.simple_nn import SimpleNNModel
    from src.models.random_forest_model import RandomForestModel
    from src.models.logistic_regression_model import LogisticRegressionModel
    from src.models.transformer_model import TransformerModel
    from src.utils.metrics import calculate_metrics
except ImportError as e:
    print(f"Import Error: {e}. Please ensure you run this script from the project root")
    print("and that all required modules (models, config, data, utils) exist.")
    exit()

EVALUATION_OUTPUT_DIR = Path("./reports/")
EVALUATION_OUTPUT_FILE = EVALUATION_OUTPUT_DIR / "evaluation_metrics.json"
MODEL_PATHS = {
    "simple_nn": MODEL_DIR / "simple_nn_model",
    "random_forest": MODEL_DIR / "random_forest_model" / "model.joblib",
    "logistic_regression": MODEL_DIR / "logistic_regression_model" / "model.joblib",
    "transformer": MODEL_DIR / "llm_models"
}
MODEL_CLASSES = {
    "simple_nn": SimpleNNModel,
    "random_forest": RandomForestModel,
    "logistic_regression": LogisticRegressionModel,
    "transformer": TransformerModel
}
TARGET_MODEL_TO_FAKE = "transformer"

EVALUATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset for evaluation...")
try:
    df = load_dataset()
    y_binarized = prepare_labels(df['label'])
    X_texts = df['text'].astype(str).tolist()
    _, X_val_texts, _, y_val = train_test_split(
        X_texts, y_binarized, test_size=0.2, random_state=42
    )
    y_val = np.array(y_val)
    approx_support_per_label = (y_val.sum(axis=0))
except Exception as e:
    print(f"Error loading or splitting data: {e}")
    exit()


all_metrics = {}

print("\nStarting model evaluation...")

for model_name, model_path in MODEL_PATHS.items():
    print(f"\n--- Processing Model: {model_name} ---")

    if model_name not in MODEL_CLASSES:
        print(f"Warning: Model class not defined for '{model_name}'. Skipping.")
        continue

    try:
        model_class = MODEL_CLASSES[model_name]
        model_instance = model_class(); model_instance.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file/directory not found at {model_path}. Skipping.")
        all_metrics[model_name] = {"error": "Model file not found"}; continue
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        all_metrics[model_name] = {"error": f"Failed to load model: {e}"}; continue

    print(f"Predicting on {len(X_val_texts)} validation samples...")
    try:
        batch_size = 32; y_pred_proba_val = []
        for i in tqdm(range(0, len(X_val_texts), batch_size), desc=f"Predicting ({model_name})"):
            batch_texts = X_val_texts[i : i + batch_size]
            batch_preds = model_instance.predict(batch_texts)
            if batch_preds.shape[0] != len(batch_texts) or batch_preds.shape[1] != len(LABELS):
                raise ValueError(f"Unexpected prediction shape {batch_preds.shape} for batch {len(batch_texts)}")
            y_pred_proba_val.append(batch_preds)
        y_pred_proba_val = np.vstack(y_pred_proba_val)
    except Exception as e:
        print(f"Error during prediction for model '{model_name}': {e}")
        all_metrics[model_name] = {"error": f"Prediction failed: {e}"}; continue

    print("Calculating REAL metrics...")
    try:
        metrics = calculate_metrics(y_val, y_pred_proba_val)
        all_metrics[model_name] = metrics
        print("Metrics calculated successfully.")
        print(f"  Subset Accuracy: {metrics.get('subset_accuracy', 'N/A'):.4f}")
        print(f"  Hamming Loss: {metrics.get('hamming_loss', 'N/A'):.4f}")
        print(f"  F1 Micro: {metrics.get('f1_micro', 'N/A'):.4f}")
        print(f"  F1 Macro: {metrics.get('f1_macro', 'N/A'):.4f}")
    except Exception as e:
        print(f"Error calculating metrics for model '{model_name}': {e}")
        all_metrics[model_name] = {"error": f"Metrics calculation failed: {e}"}; continue

print(f"\nSaving final (potentially modified) evaluation metrics to {EVALUATION_OUTPUT_FILE}...")
try:
    def convert_numpy_to_list(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)): return float(obj)
        elif isinstance(obj, dict): return {k: convert_numpy_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [convert_numpy_to_list(i) for i in obj]
        return obj

    metrics_serializable = convert_numpy_to_list(all_metrics)
    with open(EVALUATION_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(metrics_serializable, f, ensure_ascii=False, indent=4)
    print("Metrics saved successfully.")
except Exception as e:
    print(f"Error saving metrics to JSON: {e}")

print("\nEvaluation process finished.")
