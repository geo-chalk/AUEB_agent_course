"""Fake settings for every test: pytest needs no .env, contacts nothing and writes no log file."""

import pytest


@pytest.fixture(autouse=True)
def fake_settings(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://fake")
    monkeypatch.setenv("LLM_API_KEY", "fake")
    monkeypatch.setenv("LLM_MODEL", "fake-model")
    monkeypatch.setenv("EMBEDDING_MODEL", "")
    monkeypatch.setenv("LOG_DIR", "")  # stdout only: no logs/app.log from the tests
