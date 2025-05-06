import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from src.models.base_model import BaseModel
from src.config.config import NUM_CLASSES, RF_MAX_FEATURES
import numpy as np

class RandomForestModel(BaseModel):

    def __init__(self, num_classes=NUM_CLASSES, max_features=RF_MAX_FEATURES, random_state=42):
        self.num_classes = num_classes
        self.max_features = max_features
        self.random_state = random_state
        self.model = self.build_model(class_weight='balanced_subsample')

    def build_model(self, class_weight=None):
        vectorizer = TfidfVectorizer(max_features=self.max_features)

        rf_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight=class_weight
        )
        multi_output_rf = MultiOutputClassifier(rf_classifier, n_jobs=-1)

        pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('clf', multi_output_rf)
        ])
        return pipeline

    def train(self, train_texts, train_labels, val_texts=None, val_labels=None):
        print("Training RandomForest model...")
        self.model.fit(train_texts, train_labels)
        print("Training complete.")
        if val_texts is not None and val_labels is not None:
            try:
                score = self.model.score(val_texts, val_labels)
                print(f"Validation Accuracy (subset accuracy): {score:.4f}")
            except Exception as e:
                print(f"Could not score RandomForest on validation set: {e}")

    def predict(self, texts):
        if not isinstance(texts, (list, np.ndarray)):
            texts = [texts]

        try:
            proba_list = self.model.predict_proba(texts)
            proba_positive = np.array([proba[:, 1] for proba in proba_list]).T

            if proba_positive.ndim == 1 and len(texts) == 1:
                proba_positive = proba_positive.reshape(1, -1)
            elif proba_positive.ndim == 1 and len(texts) > 1:
                print(f"Warning: Unexpected RF predict_proba output shape {proba_positive.shape} for {len(texts)} inputs.")
                proba_positive = proba_positive.reshape(len(texts), -1)

            return proba_positive.astype(float)

        except AttributeError:
            print("Warning: predict_proba not available for RF, falling back to predict.")
            predictions = self.model.predict(texts)
            return predictions.astype(float)
        except Exception as e:
            print(f"Error during RandomForest predict_proba: {e}")
            return np.zeros((len(texts), self.num_classes), dtype=float)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump(self.model, path)
            print(f"RandomForest model saved to {path}")
        except Exception as e:
            print(f"Error saving RandomForest model to {path}: {e}")
            raise

    def load(self, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"RandomForest model file not found at {path}")
        try:
            self.model = joblib.load(path)
            print(f"RandomForest model loaded from {path}")
        except Exception as e:
            print(f"Error loading RandomForest model from {path}: {e}")
            raise