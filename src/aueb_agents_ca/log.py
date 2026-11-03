"""Logging setup: one line per event, each tagged with a request id so the lines of one request can be
found together even when requests overlap. Logs go to stdout, which is where Docker collects them."""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar

# Set by the API for the duration of a request; "-" outside a request (notebooks, scripts, tests).
request_id: ContextVar[str] = ContextVar("request_id", default="-")

FORMAT = "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"


class RequestIdFilter(logging.Filter):
    """Copies the current request id onto every record, so the format string can print it."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id.get()
        return True


_handler: logging.Handler | None = None  # the one handler this module installs


def configure_logging(level: str = "INFO") -> None:
    """Send our log lines to stdout in a single format. Safe to call more than once (notebook re-runs):
    it replaces only its own handler and leaves any others (pytest's, for example) alone."""
    global _handler
    root = logging.getLogger()
    if _handler is not None:
        root.removeHandler(_handler)
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter(FORMAT))
    _handler.addFilter(RequestIdFilter())
    root.addHandler(_handler)
    root.setLevel(level.upper())
