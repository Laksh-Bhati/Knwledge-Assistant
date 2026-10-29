import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("rag_app")
logger.setLevel(logging.INFO)
logger.propagate = False

# Avoid adding duplicate handlers if Streamlit reloads the app.
if not logger.handlers:
    log_file = LOG_DIR / "app.log"

    handler = RotatingFileHandler(
        log_file,
        maxBytes=1_000_000,  # start a new file at about 1 MB
        backupCount=3,
        encoding="utf-8",
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
