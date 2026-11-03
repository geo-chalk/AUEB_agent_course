"""Logging setup: one line per event, each tagged with a request id so the lines of one request can be
found together even when requests overlap. Lines go to stdout, which is where Docker collects them, and to
a rotating file under logs/ for reading after the fact."""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path
from queue import Queue

# Set by the API for the duration of a request; "-" outside a request (notebooks, scripts, tests).
request_id: ContextVar[str] = ContextVar("request_id", default="-")

FORMAT = "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"

# Libraries that log every HTTP request at INFO: httpx (our page fetches) and httpx2 (the copy inside the
# openai SDK). Useful when debugging, noise otherwise, so they are shown only at DEBUG.
CHATTY_LIBRARIES = ("httpx", "httpx2", "httpcore", "httpcore2", "primp")

# The log file rotates: when app.log reaches MAX_BYTES it becomes app.log.1 and a fresh app.log starts,
# and the oldest of the BACKUP_COUNT rotated files is deleted. Five files of 5 MB at most on disk.
LOG_FILE = "app.log"
MAX_BYTES = 0.5 * 1024 * 1024
BACKUP_COUNT = 4


class RequestIdFilter(logging.Filter):
    """Copies the current request id onto every record, so the format string can print it."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id.get()
        return True


_handlers: list[logging.Handler] = []  # the handlers this module installed, so it can replace them


def configure_logging(*, level: str = "INFO", log_dir: str | Path | None = "logs") -> None:
    """Send our log lines to stdout and, unless log_dir is None, to a rotating file in log_dir, in one
    format. Safe to call more than once (notebook re-runs): it replaces only its own handlers and leaves
    any others (pytest's, for example) alone."""
    global _handlers
    root = logging.getLogger()
    for handler in _handlers:
        root.removeHandler(handler)
        handler.close()

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]  # stdout: what Docker collects
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                Path(log_dir) / LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
            )
        )
    for handler in handlers:
        handler.setFormatter(logging.Formatter(FORMAT))
        handler.addFilter(RequestIdFilter())
        root.addHandler(handler)
    _handlers = handlers
    root.setLevel(level.upper())

    # we want to silence libraries that generate a log of logs, or turn them to debug
    for name in CHATTY_LIBRARIES:
        logging.getLogger(name).setLevel(logging.DEBUG if level.upper() == "DEBUG" else logging.WARNING)


class RequestLogQueue(logging.Handler):
    """Collects the INFO lines of one request into a queue, so they can be streamed to the client while the
    request is still running. Attach to the root logger for the duration of the request, then remove."""

    def __init__(self, *, rid: str) -> None:
        super().__init__(level=logging.INFO)
        self.rid = rid
        self.lines: Queue[str] = Queue()
        self.setFormatter(logging.Formatter("%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S"))

    def filter(self, record: logging.LogRecord) -> bool:
        return request_id.get() == self.rid  # only this request's lines, even with other requests running

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.put(self.format(record))
