"""Logging set-up on top of the standard `logging` module."""

from __future__ import annotations

import logging
from pathlib import Path

ROOT_LOGGER_NAME = "jardik"
_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_HANDLER_TAG = "_jardik_handler"


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    """Configure the `jardik` logger. Safe to call several times (handlers are replaced)."""
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(level.upper())
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_TAG, False):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(_FORMAT)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    for handler in handlers:
        handler.setFormatter(formatter)
        setattr(handler, _HANDLER_TAG, True)
        logger.addHandler(handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child of the `jardik` logger, e.g. get_logger('dataset_manager.matching')."""
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
