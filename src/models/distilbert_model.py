import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
from src.models.base_model import BaseModel


class DistilBERTModel(BaseModel):

    def __init__(self, num_classes, model_name="distilbert-base-multilingual-cased"):
        self.num_classes = num_classes
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = self.build_model()

    def build_model(self):
        bert = TFAutoModel.from_pretrained(self.model_name)

        input_ids = tf.keras.layers.Input(shape=(2,), dtype=tf.int32, name="input_ids")
        outputs = bert(input_ids)[0][:, 0, :]  # берем только CLS токен
        outputs = tf.keras.layers.Dense(2, activation='relu')(outputs)
        outputs = tf.keras.layers.Dropout(0.2)(outputs)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='sigmoid')(outputs)

        model = tf.keras.models.Model(inputs=input_ids, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, train_data, val_data):
        self.model.fit(train_data, validation_data=val_data, epochs=1)

    def predict(self, texts):
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=20, return_tensors="tf")
        preds = self.model.predict(encodings['input_ids'])
        return preds

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)

