"""Logging setup utilities.

Provides a single factory function to create configured `logging.Logger`
instances, avoiding module-level global logger state.
"""

from __future__ import annotations

import logging
from pathlib import Path

from iris_mlops.config import LoggingSectionConfig


def get_logger(name: str, config: LoggingSectionConfig) -> logging.Logger:
    """Create (or retrieve) a configured logger.

    Args:
        name: Logger name, typically ``__name__`` of the caller.
        config: Logging configuration section describing level, format,
            and output destinations.

    Returns:
        A configured `logging.Logger` instance with console and/or file
        handlers attached exactly once.
    """
    logger = logging.getLogger(name)
    logger.setLevel(config.level)

    if not logger.handlers:
        formatter = logging.Formatter(fmt=config.format, datefmt=config.date_format)

        if config.console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        if config.file:
            log_dir = Path(config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_dir / config.log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    logger.propagate = False
    return logger
