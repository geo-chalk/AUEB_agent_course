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
        health = client.get("/health").json()
        assert health["status"] == "ok" and health["model"] == "fake-model"
        response = client.post("/summarise", json={"product": "XM5"})

    app.dependency_overrides.clear()
    assert response.status_code == 200, response.text
    assert response.json() == {"product": "XM5", "summary": "a summary", "model": "fake-model", "steps": 2}
    assert len(response.headers["X-Request-ID"]) == 8  # the id that tags this request's log lines


def test_summarise_validates_its_input():
    with TestClient(app) as client:
        assert client.post("/summarise", json={}).status_code == 422  # product is required
        assert client.post("/summarise", json={"product": "XM5", "max_steps": 0}).status_code == 422


def test_summarise_logs_what_happened(monkeypatch, caplog):
    monkeypatch.setenv("LLM_BASE_URL", "http://fake")
    monkeypatch.setenv("LLM_API_KEY", "fake")
    monkeypatch.setenv("LLM_MODEL", "fake-model")
    app.dependency_overrides[get_tools] = lambda: FAKE_TOOLS
    with TestClient(app) as client, caplog.at_level("INFO"):
        client.app.state.llm = ScriptedLLM([message(content="a summary")])
        client.post("/summarise", json={"product": "XM5"})
    app.dependency_overrides.clear()

    assert "agent task='Summarise the reviews of the XM5.'" in caplog.text
    assert "agent done: 1 model calls" in caplog.text


def test_ask_runs_the_agent_on_a_free_form_task(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://fake")
    monkeypatch.setenv("LLM_API_KEY", "fake")
    monkeypatch.setenv("LLM_MODEL", "fake-model")
    app.dependency_overrides[get_tools] = lambda: FAKE_TOOLS
    with TestClient(app) as client:
        client.app.state.llm = ScriptedLLM([message(content="Reviewers dislike the price.")])
        response = client.post("/ask", json={"task": "What do reviewers dislike about the XM5?"})
        health = client.get("/health").json()
    app.dependency_overrides.clear()

    assert response.status_code == 200, response.text
    assert response.json()["answer"] == "Reviewers dislike the price." and response.json()["steps"] == 1
    assert health["tools"] == ["search_reviews", "analyze_pages"]
