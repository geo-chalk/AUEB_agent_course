"""The agent as a web service: one endpoint that runs the loop, one that says whether the service is up."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from ddgs import DDGS
from fastapi import Depends, FastAPI, Request, Response
from openai import OpenAI
from pydantic import BaseModel, Field
from starlette.responses import RedirectResponse

from aueb_agents_ca.agent import run_agent
from aueb_agents_ca.log import configure_logging, request_id
from aueb_agents_ca.settings import Settings
from aueb_agents_ca.tools import ReviewTools

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Runs once at startup: read the settings and create the clients that every request will share."""
    # read the settings
    app.state.settings = Settings.from_env()
    configure_logging(app.state.settings.log_level, app.state.settings.log_dir)
    logger.info("startup: model=%s base_url=%s", app.state.settings.model, app.state.settings.base_url)

    # define classes needed to run the Agent
    app.state.llm = OpenAI(
        base_url=app.state.settings.base_url, api_key=app.state.settings.api_key, timeout=60, max_retries=3
    )
    app.state.http = httpx.Client(follow_redirects=True, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    app.state.search = DDGS()
    yield
    app.state.http.close()  # runs once at shutdown
    logger.info("shutdown")


# the app is defined outside any function
app = FastAPI(title="Review summariser", lifespan=lifespan)


# --- Pyndatic classes are important when working with FastAPI
class SummariseRequest(BaseModel):
    product: str = Field(examples=["Sony WH-1000XM5"])
    max_steps: int = Field(default=8, ge=1, le=20)


class SummariseResponse(BaseModel):
    product: str
    summary: str
    model: str
    steps: int  # how many model calls the loop made


@app.middleware("http")
async def tag_request(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Give every request a short id: every log line it produces carries it, and so does the response."""
    rid = uuid.uuid4().hex[:8]
    token = request_id.set(rid)
    try:
        response = await call_next(request)
    finally:
        request_id.reset(token)
    response.headers["X-Request-ID"] = rid
    return response


def get_tools(request: Request) -> ReviewTools:
    """A fresh ReviewTools per request (its page cache belongs to one run), built from the shared clients."""
    state = request.app.state
    return ReviewTools(llm=state.llm, settings=state.settings, http=state.http, search=state.search)


@app.get("/", include_in_schema=False)
async def docs_redirect():
    """Redirect the landing page to docs so we can see the endpoints"""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health(request: Request) -> dict:
    """Cheap check that the service is up and configured; it makes no model call."""
    return {
        "status": "ok",
        "model": request.app.state.settings.model,
        "embedding_model": request.app.state.settings.embedding_model,
    }


@app.post("/summarise", response_model=SummariseResponse)
def summarise(
    body: SummariseRequest, request: Request, tools: Annotated[ReviewTools, Depends(get_tools)]
) -> SummariseResponse:
    """Run the agent for one product. Costs about twenty model requests on the free tier."""
    state = request.app.state
    logger.info("summarise: product=%r max_steps=%d", body.product, body.max_steps)
    t0 = time.perf_counter()
    summary, conversation = run_agent(
        f"Summarise the reviews of the {body.product}.",
        state.llm,
        tools,
        state.settings.model,
        body.max_steps,
    )
    steps = sum(1 for m in conversation if not isinstance(m, dict))  # the model's messages = model calls
    logger.info("summarise: done in %.1fs, %d model calls", time.perf_counter() - t0, steps)
    return SummariseResponse(product=body.product, summary=summary, model=state.settings.model, steps=steps)
