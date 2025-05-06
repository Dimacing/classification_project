from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

from src.data.dataset import load_dataset, prepare_labels, TextDataset
from src.models.simple_nn import SimpleNNModel
from src.models.random_forest_model import RandomForestModel
from src.models.logistic_regression_model import LogisticRegressionModel
from src.config.config import MODEL_DIR, BATCH_SIZE, MAX_LEN, NUM_CLASSES, EPOCHS, RF_MAX_FEATURES

MODEL_DIR = Path(MODEL_DIR)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")
df = load_dataset()
y = prepare_labels(df['label'])
X_texts = df['text'].astype(str).tolist()
X_train_texts, X_val_texts, y_train, y_val = train_test_split(
    X_texts, y, test_size=0.2, random_state=42
)
y_train = np.array(y_train)
y_val = np.array(y_val)

def train_simple_nn():
    print("\n--- Training Simple NN ---")
    print("Adapting TextVectorization layer...")
    text_vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=RF_MAX_FEATURES,
        output_mode='tf-idf'
    )
    text_vectorizer.adapt(X_train_texts)
    print(f"Adaptation complete. Vocabulary size: {text_vectorizer.vocabulary_size()}")

    vectorizer_path = MODEL_DIR / "simple_nn_vectorizer.pkl"
    vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
    vectorizer_data = {
        'config': text_vectorizer.get_config(),
        'weights': text_vectorizer.get_weights()
    }
    joblib.dump(vectorizer_data, vectorizer_path)
    print(f"TextVectorization layer saved to {vectorizer_path}")

    model = SimpleNNModel(
        num_classes=NUM_CLASSES,
        text_vectorizer=text_vectorizer
    )
    model.build_model()
    model.model.summary(line_length=120)

    train_dataset = TextDataset(X_train_texts, y_train, batch_size=BATCH_SIZE)
    val_dataset = TextDataset(X_val_texts, y_val, batch_size=BATCH_SIZE)

    print("Starting SimpleNN training...")
    model.train(train_dataset, val_dataset)
    print("SimpleNN training complete.")

    model_save_path_nn = MODEL_DIR / "simple_nn_model"
    model.save(model_save_path_nn)

def train_random_forest():
    print("\n--- Training Random Forest ---")
    rf_model = RandomForestModel(
        num_classes=NUM_CLASSES,
        max_features=RF_MAX_FEATURES
    )
    rf_model.train(X_train_texts, y_train, X_val_texts, y_val)

    model_dir_rf = MODEL_DIR / "random_forest_model"
    model_dir_rf.mkdir(parents=True, exist_ok=True)
    model_save_path_rf = model_dir_rf / "model.joblib"
    rf_model.save(model_save_path_rf)

def train_logistic_regression():
    print("\n--- Training Logistic Regression ---")
    logreg_model = LogisticRegressionModel(
        num_classes=NUM_CLASSES,
        max_features=RF_MAX_FEATURES
    )
    logreg_model.train(X_train_texts, y_train, X_val_texts, y_val)

    model_dir_logreg = MODEL_DIR / "logistic_regression_model"
    model_dir_logreg.mkdir(parents=True, exist_ok=True)
    model_save_path_logreg = model_dir_logreg / "model.joblib"
    logreg_model.save(model_save_path_logreg)

if __name__ == "__main__":
    print("Starting model training process...")

    train_simple_nn()
    train_random_forest()
    train_logistic_regression()

    print("\nAll models trained and saved.")