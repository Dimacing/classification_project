import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.utils import Sequence
import numpy as np
from src.config.config import LABELS, DATA_DIR

class TextDataset(Sequence):
    def __init__(self, texts, labels, batch_size=32, tokenizer=None, max_len=128):
        self.texts = texts
        self.labels = labels
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return int(np.ceil(len(self.texts) / self.batch_size))

    def __getitem__(self, idx):
        batch_start = idx * self.batch_size
        batch_end = (idx + 1) * self.batch_size

        batch_texts = self.texts[batch_start:batch_end]
        batch_labels = self.labels[batch_start:batch_end]
        if self.tokenizer:
            raise NotImplementedError("Tokenizer usage is not currently implemented in TextDataset getitem")
        else:
            return np.array(batch_texts), np.array(batch_labels)


def load_dataset():
    topics = LABELS
    files = ["5.csv", "6.csv", "1.csv", "4.csv", "3.csv", "2.csv"]

    if len(topics) != len(files):
        raise ValueError("Mismatch between number of topics and files")

    data = []
    labels = []
    found_files = 0

    print(f"Looking for data files in: {DATA_DIR.resolve()}")

    for topic, file_name in zip(topics, files):
        file_path = DATA_DIR / file_name
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                df['text'] = df[['doc_text', 'image2text', 'speech2text']].fillna('').astype(str).agg(' '.join, axis=1)
                df['text'] = df['text'].str.strip()
                df = df[df['text'] != '']

                for text in df['text']:
                    data.append(text)
                    labels.append([topic])
                print(f"Loaded {len(df)} samples from {file_name} for topic '{topic}'")
                found_files += 1
            except Exception as e:
                print(f"Error reading or processing {file_path}: {e}")
        else:
            print(f"Warning: File not found - {file_path}")

    if not data:
        raise FileNotFoundError(f"No data loaded. Check if CSV files exist in {DATA_DIR}")

    print(f"Total files processed: {found_files}/{len(files)}")
    print(f"Total samples loaded: {len(data)}")

    df_final = pd.DataFrame({'text': data, 'label': labels})
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_final


def prepare_labels(labels_list):
    mlb = MultiLabelBinarizer(classes=LABELS)
    y = mlb.fit_transform(labels_list)
    print(f"Labels prepared. Shape: {y.shape}. Classes: {mlb.classes_}")
    return y