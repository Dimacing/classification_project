from pathlib import Path

DATA_DIR = Path("./data/")
MODEL_DIR = Path("./models/")
LABELS = ["спорт", "юмор", "реклама", "соцсети", "политика", "личная жизнь"]
NUM_CLASSES = len(LABELS)
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5
