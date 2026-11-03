"""The logging setup: our lines at the chosen level, a rotating file when asked, library chatter only at DEBUG."""

import logging
from logging.handlers import RotatingFileHandler

from aueb_agents_ca.log import BACKUP_COUNT, MAX_BYTES, configure_logging


def test_http_libraries_are_quiet_unless_debug():
    configure_logging("INFO", log_dir=None)
    assert logging.getLogger("httpx2").getEffectiveLevel() == logging.WARNING
    assert logging.getLogger("aueb_agents_ca.agent").getEffectiveLevel() == logging.INFO

    configure_logging("DEBUG", log_dir=None)
    assert logging.getLogger("httpx2").getEffectiveLevel() == logging.DEBUG


def test_log_file_is_written_and_rotates(tmp_path):
    configure_logging("INFO", log_dir=tmp_path)
    logging.getLogger("aueb_agents_ca.test").info("hello from the test")

    file_handlers = [h for h in logging.getLogger().handlers if isinstance(h, RotatingFileHandler)]
    assert len(file_handlers) == 1
    assert (file_handlers[0].maxBytes, file_handlers[0].backupCount) == (MAX_BYTES, BACKUP_COUNT)
    assert "INFO [-] aueb_agents_ca.test: hello from the test" in (tmp_path / "app.log").read_text()

    configure_logging("INFO", log_dir=None)  # release the file and leave the process as other tests expect
    assert not [h for h in logging.getLogger().handlers if isinstance(h, RotatingFileHandler)]
