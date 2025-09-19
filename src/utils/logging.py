"""Shared logging utilities for the image repair project."""
from __future__ import annotations

import logging
from typing import Optional


def setup_logger(name: str = "image_repair", level: int = logging.INFO) -> logging.Logger:
    """Create or retrieve a module-level logger with a console handler."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def set_verbosity(logger: logging.Logger, level: int) -> None:
    """Set logger verbosity and update handler levels."""
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
