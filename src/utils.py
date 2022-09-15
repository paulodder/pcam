from decouple import config
from pathlib import Path

DDIR = Path(config("DATA_DIR"))
