from pathlib import Path

BASE_DIR = Path(r"D:\ai-image-detector")
if not BASE_DIR.exists():                       # fallback (e.g. Google Colab)
    BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = 32
BATCH_SIZE = 128
NUM_WORKERS = 2        # set to 0 if you get DataLoader errors on Windows