import tensorflow as tf
from src.models.base_model import BaseModel
from pathlib import Path
import joblib
import numpy as np
from src.config.config import EPOCHS

class SimpleNNModel(BaseModel):
    def __init__(self, num_classes=None, text_vectorizer=None):
        self.num_classes = num_classes
        self.text_vectorizer = text_vectorizer
        self.model = None

    def build_model(self):
        if self.text_vectorizer is None:
            raise ValueError("text_vectorizer must be provided to build the model")
        if self.num_classes is None:
            raise ValueError("num_classes must be set")

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1,), dtype=tf.string, name="text_input"),
            self.text_vectorizer,
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.num_classes, activation='sigmoid', name="output")
        ], name="SimpleNN")

        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            metrics=['accuracy', tf.keras.metrics.AUC(multi_label=True, name='auc')]
        )
        self.model = model
        print("SimpleNN model built and compiled.")

    def load(self, path):
        path = Path(path)
        vectorizer_path = path.parent / "simple_nn_vectorizer.pkl"

        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")
        if not vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")

        try:
            self.model = tf.keras.models.load_model(str(path), compile=False)

            vectorizer_data = joblib.load(vectorizer_path)
            self.text_vectorizer = tf.keras.layers.TextVectorization.from_config(vectorizer_data['config'])
            self.text_vectorizer.set_weights(vectorizer_data['weights'])

            loaded_config = self.model.get_config()
            inputs = tf.keras.Input(shape=(1,), dtype=tf.string, name="text_input")
            vectorized_inputs = self.text_vectorizer(inputs)
            x = vectorized_inputs
            for layer_config in loaded_config["layers"][2:]:
                layer = tf.keras.layers.deserialize(layer_config)
                x = layer(x)
            new_model = tf.keras.Model(inputs=inputs, outputs=x, name="SimpleNN_rebuilt")

            new_model.set_weights(self.model.get_weights())

            new_model.compile(
                loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                metrics=['accuracy', tf.keras.metrics.AUC(multi_label=True, name='auc')]
            )
            self.model = new_model

            self.num_classes = self.model.layers[-1].output_shape[-1]
            print("SimpleNN model and TextVectorization layer loaded successfully.")
            self.model.summary(line_length=120)

        except Exception as e:
            print(f"Error loading SimpleNN model or vectorizer: {e}")
            raise

    def train(self, train_data, val_data):
        if self.model is None:
            print("Model not built. Building model before training.")
            self.build_model()

        print(f"Starting training for {EPOCHS} epochs...")
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=EPOCHS,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_auc',
                    patience=3,
                    mode='max',
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=2,
                    min_lr=1e-6,
                    verbose=1
                )
            ],
            verbose=1
        )
        print("Training complete.")
        return history

    def predict(self, texts):
        if self.model is None:
            raise ValueError("Model has not been loaded or built yet.")
        if self.text_vectorizer is None:
            raise ValueError("Text vectorizer is not available in the loaded model.")

        if isinstance(texts, list):
            if texts and isinstance(texts[0], list):
                texts = [item[0] for item in texts if item]
            texts_np = np.array(texts, dtype=object)
        elif not isinstance(texts, np.ndarray):
            texts_np = np.array([texts], dtype=object)
        else:
            texts_np = texts

        if texts_np.ndim > 1:
            if texts_np.shape[1] == 1:
                texts_np = texts_np.flatten()
            else:
                raise ValueError(f"Input text array has unexpected shape: {texts_np.shape}. Expected (N,).")

        print(f"Predicting with SimpleNN for {len(texts_np)} samples.")
        try:
            predictions = self.model.predict(texts_np)
            return predictions
        except Exception as e:
            print(f"Error during SimpleNN prediction: {e}")
            raise

    def save(self, path):
        if self.model is None:
            raise ValueError("Cannot save a model that hasn't been built or loaded.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path), save_format="tf")
        print(f"SimpleNN model saved to {path}")