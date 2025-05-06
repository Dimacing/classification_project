from pathlib import Path

PROJECT_ROOT = Path(".")

DATA_DIR = PROJECT_ROOT / "data/"
MODEL_DIR = PROJECT_ROOT / "models/"
LABELS = ["спорт", "юмор", "реклама", "соцсети", "политика", "личная жизнь"]
NUM_CLASSES = len(LABELS)

MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5

RF_MAX_FEATURES = 10000