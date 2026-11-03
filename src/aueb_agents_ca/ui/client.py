"""The only place the UI talks to the API. The pages know nothing about the agent: they ask and display."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator

import httpx

# An agent run takes a while (search, fetch, one model call per page); do not give up early.
AGENT_TIMEOUT = 300


def api_url() -> str:
    """Where the API is. Read when needed, not at import, so the environment can change (compose, tests)."""
    return os.environ.get("API_URL", "http://localhost:8000")  # inside compose: http://api:8000


def make_client() -> httpx.Client:
    return httpx.Client(base_url=api_url())


def api_health(client: httpx.Client) -> tuple[dict | None, float]:
    """The API's /health and how long it took in milliseconds, or (None, elapsed) when it cannot be reached."""
    t0 = time.perf_counter()
    try:
        data = client.get("/health", timeout=5).json()
    except httpx.HTTPError:
        data = None
    return data, (time.perf_counter() - t0) * 1000


def summarise(client: httpx.Client, product: str, max_steps: int = 8) -> tuple[dict, str]:
    """POST /summarise. Returns the response body and the request id the API tagged the call with."""
    response = client.post(
        "/summarise", json={"product": product, "max_steps": max_steps}, timeout=AGENT_TIMEOUT
    )
    response.raise_for_status()
    return response.json(), response.headers.get("X-Request-ID", "-")


def ask(client: httpx.Client, task: str, max_steps: int = 8) -> tuple[dict, str]:
    """POST /ask. Returns the response body and the request id."""
    response = client.post("/ask", json={"task": task, "max_steps": max_steps}, timeout=AGENT_TIMEOUT)
    response.raise_for_status()
    return response.json(), response.headers.get("X-Request-ID", "-")


def stream_summarise(client: httpx.Client, product: str, max_steps: int = 8) -> Iterator[dict]:
    """POST /summarise/stream and yield each line as it arrives: {"log": ...} while the agent runs, then
    {"result": ...} or {"error": ...}. Every event also carries the request id."""
    with client.stream(
        "POST", "/summarise/stream", json={"product": product, "max_steps": max_steps}, timeout=AGENT_TIMEOUT
    ) as response:
        response.raise_for_status()
        rid = response.headers.get("X-Request-ID", "-")
        for line in response.iter_lines():
            if line:
                yield {**json.loads(line), "request_id": rid}
