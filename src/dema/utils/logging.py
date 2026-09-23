"""Per-case file logging."""

from __future__ import annotations

import logging
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def get_console_logger(name: str = "dema") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


class CaseLogger:
    """Context manager writing one log file per (method, dataset, case)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.logger: logging.Logger | None = None
        self._handler: logging.Handler | None = None

    def __enter__(self) -> logging.Logger:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        name = f"dema.case.{self.path}"
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        handler = logging.FileHandler(self.path, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
        self.logger, self._handler = logger, handler
        return logger

    def __exit__(self, *exc) -> None:
        if self.logger is not None and self._handler is not None:
            self.logger.removeHandler(self._handler)
            self._handler.close()
