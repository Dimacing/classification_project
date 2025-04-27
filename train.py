from sklearn.model_selection import train_test_split
from src.data.dataset import load_dataset, prepare_labels, TextDataset
from src.models.simple_nn import SimpleNNModel
from src.models.distilbert_model import DistilBERTModel
from src.config.config import MODEL_DIR, BATCH_SIZE, MAX_LEN, NUM_CLASSES
import os
import tensorflow as tf

os.makedirs(MODEL_DIR, exist_ok=True)

def train_simple_nn():
    df = load_dataset()
    X_train, X_val, y_train, y_val = train_test_split(df['text'], prepare_labels(df['label']), test_size=0.2, random_state=42)

    # Создаем и адаптируем векторный слой
    text_vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=10000,
        output_mode='tf-idf'
    )
    text_vectorizer.adapt(X_train.tolist())

    # Передаем text_vectorizer в SimpleNNModel
    model = SimpleNNModel(
        num_classes=NUM_CLASSES,
        text_vectorizer=text_vectorizer  # Передаем здесь
    )

    # Создаем датасеты с учетом батча
    train_dataset = TextDataset(X_train.tolist(), y_train, batch_size=32)
    val_dataset = TextDataset(X_val.tolist(), y_val, batch_size=32)

    model.train(train_dataset, val_dataset)
    model.save(MODEL_DIR / "simple_nn_model")

# Обучение DistilBERT модели (оставляем как есть)
def train_distilbert():
    df = load_dataset()
    X_train, X_val, y_train, y_val = train_test_split(df['text'], prepare_labels(df['label']), test_size=0.2, random_state=42)

    model = DistilBERTModel(num_classes=NUM_CLASSES)

    tokenizer = model.tokenizer
    train_dataset = TextDataset(X_train.tolist(), y_train, batch_size=BATCH_SIZE, tokenizer=tokenizer, max_len=MAX_LEN)
    val_dataset = TextDataset(X_val.tolist(), y_val, batch_size=BATCH_SIZE, tokenizer=tokenizer, max_len=MAX_LEN)

    model.train(train_dataset, val_dataset)
    model.save(MODEL_DIR / "distilbert_model")

if __name__ == "__main__":
    print("Training Simple NN...")
    train_simple_nn()

    # print("Training DistilBERT model...")
    # train_distilbert()

    print("All models trained and saved.")
