"""The agent as a web service: one endpoint that runs the loop, one that says whether the service is up."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from ddgs import DDGS
from fastapi import Depends, FastAPI, Request
from openai import OpenAI
from pydantic import BaseModel, Field

from aueb_agents_ca.agent import run_agent
from aueb_agents_ca.settings import Settings
from aueb_agents_ca.tools import ReviewTools


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Runs once at startup: read the settings and create the clients that every request will share."""
    app.state.settings = Settings.from_env()
    app.state.llm = OpenAI(
        base_url=app.state.settings.base_url, api_key=app.state.settings.api_key, timeout=60, max_retries=3
    )
    app.state.http = httpx.Client(follow_redirects=True, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    app.state.search = DDGS()
    yield
    app.state.http.close()  # runs once at shutdown


app = FastAPI(title="Review summariser", lifespan=lifespan)


def get_tools(request: Request) -> ReviewTools:
    """A fresh ReviewTools per request (its page cache belongs to one run), built from the shared clients."""
    state = request.app.state
    return ReviewTools(llm=state.llm, settings=state.settings, http=state.http, search=state.search)


class SummariseRequest(BaseModel):
    product: str = Field(examples=["Sony WH-1000XM5"])
    max_steps: int = Field(default=8, ge=1, le=20)


class SummariseResponse(BaseModel):
    product: str
    summary: str
    model: str
    steps: int  # how many model calls the loop made


@app.get("/health")
def health(request: Request) -> dict:
    """Cheap check that the service is up and configured; it makes no model call."""
    return {"status": "ok", "model": request.app.state.settings.model}


@app.post("/summarise", response_model=SummariseResponse)
def summarise(
    body: SummariseRequest, request: Request, tools: Annotated[ReviewTools, Depends(get_tools)]
) -> SummariseResponse:
    """Run the agent for one product. Costs about twenty model requests on the free tier."""
    state = request.app.state
    summary, conversation = run_agent(
        f"Summarise the reviews of the {body.product}.",
        state.llm,
        tools,
        state.settings.model,
        body.max_steps,
    )
    steps = sum(1 for m in conversation if not isinstance(m, dict))  # the model's messages = model calls
    return SummariseResponse(product=body.product, summary=summary, model=state.settings.model, steps=steps)
