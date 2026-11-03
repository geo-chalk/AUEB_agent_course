"""The API end to end through FastAPI's test client, with the model and the tools replaced by fakes."""

from fakes import FAKE_TOOLS, ScriptedLLM, message, tool_call
from fastapi.testclient import TestClient

from aueb_agents_ca.api import app, get_tools


def test_health_and_summarise_with_fake_dependencies(monkeypatch):
    # The lifespan reads these at startup; nothing real is contacted.
    monkeypatch.setenv("LLM_BASE_URL", "http://fake")
    monkeypatch.setenv("LLM_API_KEY", "fake")
    monkeypatch.setenv("LLM_MODEL", "fake-model")
    app.dependency_overrides[get_tools] = lambda: FAKE_TOOLS  # FastAPI's own dependency injection

    with TestClient(app) as client:  # 'with' runs the lifespan (startup and shutdown)
        client.app.state.llm = ScriptedLLM(
            [message(tool_calls=[tool_call("search_reviews", query="x")]), message(content="a summary")]
        )
        assert client.get("/health").json() == {"status": "ok", "model": "fake-model"}
        response = client.post("/summarise", json={"product": "XM5"})

    app.dependency_overrides.clear()
    assert response.status_code == 200, response.text
    assert response.json() == {"product": "XM5", "summary": "a summary", "model": "fake-model", "steps": 2}


def test_summarise_validates_its_input():
    with TestClient(app) as client:
        assert client.post("/summarise", json={}).status_code == 422  # product is required
        assert client.post("/summarise", json={"product": "XM5", "max_steps": 0}).status_code == 422
