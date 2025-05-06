from pathlib import Path
import joblib
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from src.models.base_model import BaseModel
from src.config.config import LABELS

LLM_MODEL_PATH = "./llm_models"
LLM_MAX_LEN = 256
LLM_PREDICT_BATCH_SIZE = 8

class TransformerModel(BaseModel):
    def __init__(self, model_path=LLM_MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.mlb = None
        self.device = None
        self.num_labels = len(LABELS)

    def build_model(self):
        pass

    def train(self, train_data, val_data):
        raise NotImplementedError("This class loads a pre-trained transformer model, training is not implemented here.")

    def load(self, path=None):
        load_path = Path(path) if path else self.model_path
        print(f"Loading Transformer model components from {load_path}...")
        if not load_path.is_dir():
            raise FileNotFoundError(f"Transformer model directory not found: {load_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Using device: {self.device}")

        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(load_path).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(load_path)
            print("  Model and tokenizer loaded.")

            mlb_path = load_path / 'label_binarizer.pkl'
            if mlb_path.exists():
                self.mlb = joblib.load(mlb_path)
                print(f"  Label binarizer loaded from {mlb_path}")
                if set(self.mlb.classes_) != set(LABELS):
                    print(f"Warning: Loaded MLB classes {self.mlb.classes_} differ from config LABELS {LABELS}")
                self.num_labels = len(self.mlb.classes_)
                if self.model.config.num_labels != self.num_labels:
                    print(f"Warning: Model config num_labels ({self.model.config.num_labels}) != loaded MLB ({self.num_labels}).")
            else:
                print(f"Warning: Label binarizer not found at {mlb_path}. Cannot map output to labels reliably.")
                self.mlb = None

        except Exception as e:
            print(f"Error loading transformer model from {load_path}: {e}")
            raise

    def predict(self, texts):
        if self.model is None or self.tokenizer is None:
            raise ValueError("Transformer model or tokenizer not loaded. Call load() first.")
        if not isinstance(texts, list):
            texts = [texts]

        print(f"Predicting with Transformer for {len(texts)} samples...")
        try:
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=LLM_MAX_LEN,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits

            probabilities = torch.sigmoid(logits)
            probabilities_np = probabilities.cpu().numpy()

            if len(texts) == 1 and probabilities_np.shape == (self.num_labels,):
                probabilities_np = probabilities_np.reshape(1, -1)

            print(f"  Prediction output shape: {probabilities_np.shape}")
            return probabilities_np.astype(float)

        except Exception as e:
            print(f"Error during Transformer prediction: {e}")
            return np.zeros((len(texts), self.num_labels), dtype=float)

    def save(self, path):
        raise NotImplementedError("Saving is handled by the original training script.")