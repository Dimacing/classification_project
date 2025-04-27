# src/models/simple_nn.py
import tensorflow as tf
from src.models.base_model import BaseModel

class SimpleNNModel(BaseModel):

    def __init__(self, num_classes=None, text_vectorizer=None):
        self.num_classes = num_classes
        self.text_vectorizer = text_vectorizer
        self.model = None  # Model is not built immediately

    def build_model(self):
        if self.text_vectorizer is None:
            raise ValueError("text_vectorizer must be set before building the model")
        if self.num_classes is None:
            raise ValueError("num_classes must be set")

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1,), dtype=tf.string),
            self.text_vectorizer,
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(self.num_classes, activation='sigmoid')
        ])
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        self.model = model

    def load(self, path):
        self.model = tf.keras.models.load_model(str(path))
        self.text_vectorizer = self.model.layers[0]
        self.num_classes = self.model.layers[-1].output_shape[-1]

    def train(self, train_data, val_data):
        if self.model is None:
            self.build_model()
        self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=5,
            batch_size=32
        )

    def predict(self, texts):
        return self.model.predict(texts)

    def save(self, path):
        self.model.save(path, save_format="tf")