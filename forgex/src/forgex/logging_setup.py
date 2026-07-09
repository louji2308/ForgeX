from __future__ import annotations

import logging
import sys
from typing import Optional


_LOGERS: dict[str, logging.Logger] = {}


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    if name in _LOGERS:
        return _LOGERS[name]

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        ))
        logger.addHandler(handler)
        resolved_level = (level or "INFO").upper()
        logger.setLevel(getattr(logging, resolved_level, logging.INFO))
    logger.propagate = False
    _LOGERS[name] = logger
    return logger
