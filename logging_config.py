# logging_config.py
import logging
import sys

_setup_logging_done = False

def setup_logging(level=logging.INFO):
    global _setup_logging_done

    logging.basicConfig(
        level=level,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
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
