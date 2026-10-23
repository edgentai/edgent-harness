"""Structured JSON logs correlated by run ID, with redaction of sensitive fields."""
from __future__ import annotations

import json
import logging
import sys
from typing import Any, Optional

from ..core.utils import now_iso, redact


class StructuredLogger:
    def __init__(self, name: str = "harness", run_id: Optional[str] = None, stream: Any = None, level: int = logging.INFO) -> None:
        self.run_id = run_id
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._stream = stream or sys.stderr
        if not self._logger.handlers:
            h = logging.StreamHandler(self._stream)
            h.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(h)
            self._logger.propagate = False

    def _emit(self, level: int, event: str, **fields: Any) -> None:
        rec = {"ts": now_iso(), "level": logging.getLevelName(level), "event": event, "run_id": self.run_id}
        rec.update(redact(fields))
        self._logger.log(level, json.dumps(rec, ensure_ascii=False, default=str))

    def info(self, event: str, **fields: Any) -> None:
        self._emit(logging.INFO, event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._emit(logging.WARNING, event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self._emit(logging.ERROR, event, **fields)

    def debug(self, event: str, **fields: Any) -> None:
        self._emit(logging.DEBUG, event, **fields)

    def bind(self, run_id: str) -> "StructuredLogger":
        child = StructuredLogger.__new__(StructuredLogger)
        child.run_id = run_id
        child._logger = self._logger
        child._stream = self._stream
        return child


_default: Optional[StructuredLogger] = None


def get_logger() -> StructuredLogger:
    global _default
    if _default is None:
        _default = StructuredLogger()
    return _default
