from pathlib import Path

BASE_DATA_DIR = str(Path("~/data").expanduser().resolve())
MODEL_CHECKPOINT_DIR = f"{BASE_DATA_DIR}/models"
GOOGLE_FONTS_DIR = f"{BASE_DATA_DIR}/fonts"
GOOGLE_FONTS_METADATA_DIR = f"{BASE_DATA_DIR}/metadata"