"""Logging configuration."""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "audio_drift",
    log_file: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create and configure a logger.

    Args:
        name: Logger name.
        log_file: Optional path to a log file.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — force UTF-8 so Windows cp1252 consoles don't crash on
    # Unicode glyphs (│ ━ α β) used in log messages.
    stream = sys.stdout
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")  # Python 3.7+
    except (AttributeError, ValueError):
        pass
    console_handler = logging.StreamHandler(stream)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler (UTF-8 so the log file matches the console)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger