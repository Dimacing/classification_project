import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.utils import Sequence
import numpy as np
from src.config.config import LABELS


class TextDataset(Sequence):
    """Класс для подачи данных в модель."""
    def __init__(self, texts, labels, batch_size=32, tokenizer=None, max_len=128):
        self.texts = texts
        self.labels = labels
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return int(np.ceil(len(self.texts) / self.batch_size))

    def __getitem__(self, idx):
        batch_texts = self.texts[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.tokenizer:
            encodings = self.tokenizer(batch_texts, truncation=True, padding=True, max_length=self.max_len, return_tensors="np")
            return encodings['input_ids'], batch_labels
        else:
            return np.array(batch_texts), batch_labels


def load_dataset():
    """Функция для загрузки всех csv файлов в общий датафрейм."""
    topics = ["спорт", "юмор", "реклама", "соцсети", "политика", "личная жизнь"]
    files = ["5.csv", "6.csv", "1.csv", "4.csv", "3.csv", "2.csv"]

    data = []
    labels = []

    for topic, file in zip(topics, files):
        df = pd.read_csv(f"./data/{file}")
        df['text'] = df[['doc_text', 'image2text', 'speech2text']].fillna('').agg(' '.join, axis=1)
        df = df[df['text'].str.strip() != '']
        for text in df['text']:
            data.append(text)
            labels.append([topic])

    df_final = pd.DataFrame({'text': data, 'label': labels})
    return df_final


def prepare_labels(labels):
    """Бинаризация меток."""
    mlb = MultiLabelBinarizer(classes=LABELS)
    y = mlb.fit_transform(labels)
    return y
