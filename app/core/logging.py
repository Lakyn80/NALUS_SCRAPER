"""
Centralized logging configuration for the NALUS RAG pipeline.

Usage:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Something happened")

Environment variables:
    LOG_LEVEL : Python log level name, e.g. DEBUG, INFO, WARNING (default: INFO)
    LOG_FILE  : Path to a log file. If not set, logs go to stderr only.
"""

import logging
import os
import sys
from typing import Optional

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

# Module-level flag so we only configure handlers once per process.
_configured = False


def _configure_root_logger() -> None:
    global _configured
    if _configured:
        return

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    root = logging.getLogger()
    root.setLevel(level)

    # Use exact type checks — FileHandler inherits StreamHandler, so isinstance()
    # would incorrectly treat a FileHandler as a StreamHandler guard match.
    if not any(type(h) is logging.StreamHandler for h in root.handlers):
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    log_file: Optional[str] = os.getenv("LOG_FILE")
    if log_file and not any(type(h) is logging.FileHandler for h in root.handlers):
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with the project-wide configuration applied."""
    _configure_root_logger()
    return logging.getLogger(name)
