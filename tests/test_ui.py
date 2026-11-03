"""The UI is a client of the API, so it is tested against a fake API: no agent, no model, no network."""

import httpx
from streamlit.testing.v1 import AppTest

import aueb_agents_ca.ui as ui_module
from aueb_agents_ca.ui import api_health, summarise


def fake_api(handler):
    return httpx.Client(base_url="http://api", transport=httpx.MockTransport(handler))


def test_summarise_returns_the_body_and_the_request_id():
    def handler(request):
        assert request.url.path == "/summarise" and request.read() == b'{"product":"XM5"}'
        return httpx.Response(
            200, json={"summary": "# ok", "steps": 3, "model": "m"}, headers={"X-Request-ID": "abc12345"}
        )

    body, request_id = summarise(fake_api(handler), "XM5")
    assert body["summary"] == "# ok" and request_id == "abc12345"


def test_health_is_none_when_the_api_is_down():
    def handler(request):
        raise httpx.ConnectError("refused")

    assert api_health(fake_api(handler)) is None


def test_page_tells_the_user_when_the_api_is_down(monkeypatch):
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")  # nothing listens there
    app = AppTest.from_file(ui_module.__file__).run()  # the page script, wherever the package is installed
    assert "not reachable" in app.error[0].value
