"""Logging helpers for the Anytime Inference Planner."""
from __future__ import annotations

import logging
from typing import Optional


def configure_logger(name: str = "anytime_planner", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a project-wide logger.

    The helper ensures that loggers are configured exactly once even when the
    module is imported repeatedly (e.g., across experiment scripts and tests).
    """

    logger = logging.getLogger(name)
    if logger.handlers:
        logger.setLevel(level)
        return logger

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a child logger using the project configuration."""

    if name is None:
        name = "anytime_planner"
    parent = configure_logger()
    return parent.getChild(name)
