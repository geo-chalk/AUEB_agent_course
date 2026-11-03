"""The logging setup: our lines at the chosen level, library HTTP chatter only at DEBUG."""

import logging

from aueb_agents_ca.log import configure_logging


def test_http_libraries_are_quiet_unless_debug():
    configure_logging("INFO")
    assert logging.getLogger("httpx2").getEffectiveLevel() == logging.WARNING
    assert logging.getLogger("aueb_agents_ca.agent").getEffectiveLevel() == logging.INFO

    configure_logging("DEBUG")
    assert logging.getLogger("httpx2").getEffectiveLevel() == logging.DEBUG

    configure_logging("INFO")  # leave the process as the other tests expect it
