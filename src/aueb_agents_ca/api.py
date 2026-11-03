"""The agent as a web service: one endpoint that runs the loop, one that says whether the service is up."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager
from contextvars import copy_context
from queue import Empty, Queue
from threading import Thread
from typing import Annotated

import httpx
from ddgs import DDGS
from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel, Field
from starlette.responses import RedirectResponse

from aueb_agents_ca.agent import run_agent
from aueb_agents_ca.log import RequestLogQueue, configure_logging, request_id
from aueb_agents_ca.settings import Settings
from aueb_agents_ca.tools import TOOL_SCHEMAS, ReviewTools

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Runs once at startup: read the settings and create the clients that every request will share."""
    # read the settings
    app.state.settings = Settings.from_env()
    configure_logging(level=app.state.settings.log_level, log_dir=app.state.settings.log_dir)
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


class ProductImage(BaseModel):
    url: str
    thumbnail: str
    source: str
    title: str


class SummariseResponse(BaseModel):
    product: str
    summary: str
    model: str
    steps: int  # how many model calls the loop made
    image: ProductImage | None = None  # Exercise 5.4: filled from the agent's find_product_image tool


class AskRequest(BaseModel):
    task: str = Field(examples=["What do reviewers dislike about the Sony WH-1000XM5?"])
    max_steps: int = Field(default=8, ge=1, le=20)


class AskResponse(BaseModel):
    task: str
    answer: str
    model: str
    steps: int


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


def product_image(*, tools: ReviewTools, product: str) -> ProductImage | None:
    """Exercise 5.4: return the image the agent found during its run (tools.image) as a ProductImage.
    Exercise 5.5: the model sometimes skips the tool; if tools.image is empty, call it yourself first."""
    return None


def run_task(*, task: str, max_steps: int, request: Request, tools: ReviewTools) -> tuple[str, int]:
    """Run the agent on one task and return its answer and the number of model calls it made."""
    state = request.app.state
    logger.info("agent task=%r max_steps=%d", task, max_steps)
    t0 = time.perf_counter()
    answer, conversation = run_agent(
        task=task, llm=state.llm, tools=tools, model=state.settings.model, max_steps=max_steps
    )
    steps = sum(1 for m in conversation if not isinstance(m, dict))  # the model's messages = model calls
    logger.info("agent done in %.1fs, %d model calls", time.perf_counter() - t0, steps)
    return answer, steps


@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health(request: Request) -> dict:
    """Cheap check that the service is up and configured; it makes no model call."""
    settings = request.app.state.settings
    return {
        "status": "ok",
        "model": settings.model,
        "embedding_model": settings.embedding_model,
        "tools": [schema["function"]["name"] for schema in TOOL_SCHEMAS],
    }


@app.post("/summarise", response_model=SummariseResponse)
def summarise(
    body: SummariseRequest, request: Request, tools: Annotated[ReviewTools, Depends(get_tools)]
) -> SummariseResponse:
    """Summarise the reviews of one product. Costs about twenty model requests on the free tier."""
    summary, steps = run_task(
        task=f"Summarise the reviews of the {body.product}.",
        max_steps=body.max_steps,
        request=request,
        tools=tools,
    )
    return SummariseResponse(
        product=body.product,
        summary=summary,
        model=request.app.state.settings.model,
        steps=steps,
        image=product_image(tools=tools, product=body.product),
    )


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request, tools: Annotated[ReviewTools, Depends(get_tools)]) -> AskResponse:
    """Run the agent on a free-form task about a product's reviews (what the chat page sends).
    The system prompt makes the agent refuse tasks that are not about product reviews."""
    answer, steps = run_task(task=body.task, max_steps=body.max_steps, request=request, tools=tools)
    return AskResponse(task=body.task, answer=answer, model=request.app.state.settings.model, steps=steps)


@app.post("/summarise/stream")
def summarise_stream(
    body: SummariseRequest, request: Request, tools: Annotated[ReviewTools, Depends(get_tools)]
) -> StreamingResponse:
    """Like /summarise, but the response is streamed: one JSON object per line, the agent's log lines as
    they happen ({"log": ...}) and finally the result ({"result": ...}) or an error ({"error": ...}).
    A client can show what the agent is doing instead of a spinner for a minute."""
    rid = request_id.get()
    capture = RequestLogQueue(rid=rid)  # catches this request's log lines
    logging.getLogger().addHandler(capture)
    outcome: Queue[dict] = Queue()
    task = f"Summarise the reviews of the {body.product}."
    model = request.app.state.settings.model

    def work() -> None:  # the agent runs in its own thread so the response can stream meanwhile
        try:
            summary, steps = run_task(task=task, max_steps=body.max_steps, request=request, tools=tools)
            result = SummariseResponse(
                product=body.product,
                summary=summary,
                model=model,
                steps=steps,
                image=product_image(tools=tools, product=body.product),
            )
            outcome.put({"result": result.model_dump()})
        except Exception as e:
            logger.exception("agent failed")
            outcome.put({"error": str(e)})

    # copy_context: the thread keeps this request's id, so its log lines pass the handler's filter
    Thread(target=copy_context().run, args=(work,), daemon=True).start()

    def events() -> Iterator[str]:
        try:
            while True:
                try:
                    yield json.dumps({"log": capture.lines.get(timeout=0.5)}) + "\n"
                except Empty:
                    pass
                if not outcome.empty():
                    while not capture.lines.empty():  # flush the last lines before the result
                        yield json.dumps({"log": capture.lines.get()}) + "\n"
                    yield json.dumps(outcome.get()) + "\n"
                    return
        finally:
            logging.getLogger().removeHandler(capture)

    return StreamingResponse(events(), media_type="application/x-ndjson")
