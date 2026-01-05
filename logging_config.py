# logging_config.py
import logging
import sys
from pathlib import Path

LOG_PATH = Path("logs/log.db")
_setup_logging_done = False

def setup_logging(level=logging.INFO):
    global _setup_logging_done
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        logging.critical(f"No permission to create logs directory: {LOG_PATH.parent}")
        raise

    handlers = [
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ]

    logging.basicConfig(
        level=level,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

    # Reduce noise from libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    logger = logging.getLogger("app")
    _setup_logging_done = True
    return logger

def get_logger(name):
    global _setup_logging_done

    if not _setup_logging_done:
        setup_logging()

    logger = logging.getLogger(name)
    return logger
