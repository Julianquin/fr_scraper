from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROC     = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
