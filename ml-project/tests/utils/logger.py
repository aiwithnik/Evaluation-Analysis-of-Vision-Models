import logging
import warnings
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str,
    log_dir: Path,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create a logger that logs to both console and file,
    and captures Python warnings into the log file.
    """

    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger  # avoid duplicate handlers

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{name}_{timestamp}.log"

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # -------------------------
    # File handler
    # -------------------------
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    fh.setLevel(level)

    # -------------------------
    # Console handler
    # -------------------------
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(level)

    logger.addHandler(fh)
    logger.addHandler(ch)

    # -------------------------
    # CAPTURE WARNINGS
    # -------------------------
    logging.captureWarnings(True)
    warnings.simplefilter("always")

    warning_logger = logging.getLogger("py.warnings")
    warning_logger.addHandler(fh)
    warning_logger.setLevel(logging.WARNING)
    warning_logger.propagate = False

    return logger
