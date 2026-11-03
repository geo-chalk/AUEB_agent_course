"""The UI is a client of the API, so it is tested against a fake API: no agent, no model, no network."""

from pathlib import Path

import httpx
from streamlit.testing.v1 import AppTest

import aueb_agents_ca.ui as ui_package
from aueb_agents_ca.ui.client import api_health, ask, stream_summarise, summarise

UI_DIR = Path(ui_package.__file__).parent


def fake_api(handler):
    return httpx.Client(base_url="http://api", transport=httpx.MockTransport(handler))


def test_summarise_and_ask_return_the_body_and_the_request_id():
    def handler(request):
        body = {"summary": "# ok", "answer": "yes", "steps": 3, "model": "m"}
        return httpx.Response(200, json=body, headers={"X-Request-ID": "abc12345"})

    body, request_id = summarise(fake_api(handler), "XM5")
    assert body["summary"] == "# ok" and request_id == "abc12345"
    body, request_id = ask(fake_api(handler), "Is it good?")
    assert body["answer"] == "yes" and request_id == "abc12345"


def test_health_is_none_when_the_api_is_down():
    def handler(request):
        raise httpx.ConnectError("refused")

    health, latency_ms = api_health(fake_api(handler))
    assert health is None and latency_ms >= 0


def test_every_page_tells_the_user_when_the_api_is_down(monkeypatch):
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")  # nothing listens there
    for page in (UI_DIR / "Home.py", *sorted((UI_DIR / "pages").glob("*.py"))):
        app = AppTest.from_file(str(page)).run()
        assert "not reachable" in app.error[0].value, page.name


def test_stream_summarise_yields_log_lines_then_the_result():
    def handler(request):
        assert request.url.path == "/summarise/stream"
        body = b'{"log": "10:00:00 agent start"}\n{"log": "10:00:01 step 0"}\n{"result": {"summary": "# ok", "steps": 2, "model": "m"}}\n'
        return httpx.Response(200, content=body, headers={"X-Request-ID": "abc12345"})

    events = list(stream_summarise(fake_api(handler), "XM5"))
    assert [e.get("log") for e in events[:2]] == ["10:00:00 agent start", "10:00:01 step 0"]
    assert events[-1]["result"]["summary"] == "# ok" and events[-1]["request_id"] == "abc12345"
