"""Centralized logging configuration for the application.

Provides ``setup_logging`` to install file and stdout handlers once, and
``get_logger`` to retrieve named loggers (lazily triggering setup on
first use).
"""

import logging
import sys
from pathlib import Path

LOG_PATH = Path("logs/sot.log")
_setup_logging_done = False


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configures the root logger with file and stdout handlers.

    Creates the log directory if needed, installs handlers, quiets noisy
    third-party loggers, and marks setup as done so it runs only once.

    Args:
        level: The root logging level to apply.

    Returns:
        The application logger (named ``"app"``).

    Raises:
        PermissionError: If the log directory cannot be created.
    """
    global _setup_logging_done
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        logging.critical(
            f"No permission to create logs directory: {LOG_PATH.parent}"
        )
        raise

    handlers = [
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ]

    logging.basicConfig(
        level=level,
        format=("%(asctime)s | %(levelname)s | " "%(name)s | %(message)s"),
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


def get_logger(name: str) -> logging.Logger:
    """Returns a named logger, configuring logging on first use.

    Args:
        name: The logger name.

    Returns:
        The requested logger.
    """
    global _setup_logging_done

    if not _setup_logging_done:
        setup_logging()

    logger = logging.getLogger(name)
    return logger
