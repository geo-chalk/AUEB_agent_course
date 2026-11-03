# Customer Analytics: a simple review-summarizer agent

Course material for the AUEB MSc Data Science, *Advanced Customer Analytics*.

We build a small **agent** that, given a product name, searches the web for reviews,
analyses them (which aspects people discuss, with what sentiment, backed by quotes) and writes a
short, grounded summary with an LLM.
Along the way we learn four engineering tools:

| Step | Topic |
|------|-------|
| 1 | uv, repo setup, Python packages, notebooks |
| 2 | Docker for development: the notebook image |
| 3 | FastAPI, and Docker for production: the API image |
| 4 | A simple UI (Streamlit), as a second service on the production image |

Everything runs for free: the LLM is called through the Gemini free tier (or Groq, or a local Ollama),
and web search uses DuckDuckGo (no key).

## Project structure

```
aueb_agents_CA/
├── README.md
├── pyproject.toml          # project metadata and dependencies (managed by uv)
├── uv.lock                 # exact versions, so everyone installs the same thing
├── .python-version         # Python version uv will use (3.13)
├── .env.example            # template for your secrets; copy to .env
├── notebooks/              # Step 1 lessons live here (img/ holds their figures)
├── src/aueb_agents_ca/     # the package: settings, log, tools, agent, api, ui/ (read __init__.py for the map)
├── tests/                  # pytest tests that run without internet or API keys
├── logs/                   # the rotating log file lands here (ignored by git)
├── setup/Dockerfile_dev    # Step 2: the development image (notebooks)
├── setup/Dockerfile        # Step 3: the production image (the API; Step 4 runs the UI on it too)
├── docker-compose.yml      # how to run the images locally (ports, .env, mounted folders, reload)
└── .dockerignore           # what never enters the image (.env, .venv, .git)
```

## Quick start (clone and run)

```bash
git clone <this repo> && cd aueb_agents_CA
uv sync --extra api          # creates .venv and installs everything from uv.lock, including the API's packages
cp .env.example .env         # then put your Gemini key in .env
uv run pytest                # 19 tests, no internet or key needed: the setup works
uv run jupyter lab           # open notebooks/
```

[Install uv](https://docs.astral.sh/uv/getting-started/installation/) first if you do not have it.

Or, with Docker installed, skip uv entirely:

```bash
cp .env.example .env             # compose reads .env, so it must exist; put your Gemini key in it
docker compose up --build        # notebooks :8989, API :8000/docs, UI :8501
```

## Choosing an LLM provider

The code talks to any provider that implements the OpenAI chat-completions API. Pick one in `.env`:

| Provider | Key | Free limits | Notes |
|----------|-----|-------------|-------|
| **Gemini** (default) | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | `gemini-3.5-flash-lite`: 15 req/min, 250K tokens/min, 500 req/day. `gemini-embedding-001`: 100 req/min, 1K req/day, and every text in a batch counts as a request. `gemini-3.8-flash`: 5 req/min, 20 req/day | One agent run costs about 20 chat requests and up to 40 embedding requests, so a key is good for roughly 20 runs a day. Check your own numbers in AI Studio under Rate limits. Free-tier prompts may be used by Google to improve its products |
| Groq | [console.groq.com](https://console.groq.com) | 30 req/min, 8K tokens/min, 1K req/day | very fast; the agent loop can hit the tokens/min limit |
| Ollama | none | unlimited | runs on your laptop; needs 8 GB RAM; smaller models |

---

# How the repo was built, step by step

Each step below explains the files it added and the commands that produced them, so you can follow the
flow or rebuild the project yourself.

## Step 1: uv, repo setup, notebooks

### 1.1 Create the project

```bash
mkdir aueb_agents_CA && cd aueb_agents_CA
uv init --bare --python 3.13     # writes pyproject.toml
echo "3.13" > .python-version
mkdir -p notebooks src/aueb_agents_ca
touch src/aueb_agents_ca/__init__.py
```

### 1.2 Tell uv this is a package

Add this to `pyproject.toml` so `src/aueb_agents_ca` is importable from notebooks and services:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/aueb_agents_ca"]
```

### 1.3 Add dependencies

```bash
uv add openai ddgs httpx trafilatura numpy pandas python-dotenv tiktoken
uv add --group dev jupyterlab ipykernel black isort pytest
```

| Package | Why |
|---------|-----|
| `openai` | client for the OpenAI-compatible chat API (Gemini, Groq and Ollama all speak it) |
| `ddgs` | DuckDuckGo web search, no API key |
| `httpx` | download web pages |
| `trafilatura` | extract readable text from HTML |
| `numpy` | cosine similarity between embeddings |
| `pandas` | tables |
| `python-dotenv` | load `.env` |
| `tiktoken` | tokeniser, to see how text becomes tokens |

`uv add` updates `pyproject.toml`, resolves versions into `uv.lock` and installs into `.venv`.

### 1.4 Secrets and ignores

Create `.env.example` (committed) and `.env` (never committed):

```bash
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_API_KEY=AIza...
LLM_MODEL=gemini-3.5-flash-lite
EMBEDDING_MODEL=gemini-embedding-001
```

The committed `.env.example` also lists the Groq and Ollama settings, commented out.

Create `.gitignore`:

```
.venv/
.env
.idea/
.DS_Store
__pycache__/
.ipynb_checkpoints/
.pytest_cache/
```

### 1.5 Notebooks

Start Jupyter with the project environment:

```bash
uv run jupyter lab
```

| Notebook | What it teaches |
|----------|-----------------|
| `notebooks/01_llm_call.ipynb` | one chat completion; what happens inside the server; why a model on its own is not enough |
| `notebooks/02_tools.ipynb` | the tools: search, fetch page, and an aspect-based analysis (model extracts opinions, code counts them) |
| `notebooks/03_agent_loop.ipynb` | the agent loop: model asks for tools, we run them, repeat |

### 1.6 The package and its tests

The tools used by notebook 03 live in the package, `src/aueb_agents_ca/`, so that the API can reuse them
later: `settings.py` reads the configuration, `tools.py` holds the tools. They receive their dependencies
(LLM client, HTTP client, search engine, settings) from the caller, which is what lets `tests/test_tools.py`
test them with fakes, offline. Run the tests with:

```bash
uv run pytest
```

The package version of the tools is the notebook version plus the hardening a real service needs:
search failures returned as text, embedding requests batched and capped, empty pages skipped, logging
instead of prints.

## Step 2: Docker for development

The same code, running in a container: a student who has Docker needs nothing else, and "works on my
machine" becomes "works in the image".

This step builds a *development* image: Jupyter, the dev dependencies, the project folders mounted from
the laptop so edits are live, running as root. It is made for working on the code, not for shipping it.
Step 3 adds the *production* image for the API, same base and different choices, and the two side by
side are the point of the lesson.

### 2.1 Describe the image: `setup/Dockerfile_dev`

Read the file top to bottom. Every instruction is a *layer*; Docker rebuilds a
layer, and everything after it, only when its inputs change. That is why the file is ordered from the
least to the most frequently changing thing:

| Instruction | Why |
|---|---|
| `FROM python:3.13-slim` | the same Python as `.python-version`; nothing else is installed system-wide |
| `COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /bin/uv` | uv, pinned to the version on our laptops, so the build is reproducible |
| `COPY pyproject.toml uv.lock ...` then `RUN uv sync --frozen --no-install-project` | the dependencies only. Rebuilt when the lock file changes, which is rare |
| `COPY src ./src` then `RUN uv sync --frozen` | the project itself. Rebuilt on every code change, but it only links `src/` into the venv, so it costs nothing |
| `ENV PATH="/app/.venv/bin:$PATH"` | `python`, `pytest` and `jupyter` inside the container are the project's |
| `CMD ["jupyter", "lab", ...]` | what runs when the container starts. `--allow-root` because containers run as root; the empty token is fine on a laptop, never on a shared server |

The second `uv sync` is the line that is easy to forget: without it the package is never installed and
`import aueb_agents_ca` fails inside the container even though `src/` was copied.

### 2.2 Keep secrets out of the image: `.dockerignore`

```
.venv/
.git/
.env
.idea/
.pytest_cache/
__pycache__/
.ipynb_checkpoints/
```

Everything listed here is invisible to `COPY`. The key reaches the container at run time, through
compose, and is never written into an image that could be pushed somewhere.

### 2.3 Describe how to run it: `docker-compose.yml`

```yaml
services:
  notebook:
    build:
      context: .
      dockerfile: setup/Dockerfile_dev
    ports:
      - "8989:8989"                    # laptop port : container port
    env_file: .env                     # the key enters here, at run time
    volumes:
      - ./notebooks:/app/notebooks     # edit on the laptop, see it in the container
      - ./src:/app/src
      - ./tests:/app/tests
    extra_hosts:
      - "host.docker.internal:host-gateway"   # lets the container reach Ollama on the laptop (Linux)
```

Compose is the run configuration: which image, which ports, which files from the laptop are *mounted*
into the container so edits are live, and where the environment comes from. Steps 3 and 4 add an `api`
and a `ui` service to this same file.

### 2.4 Run and verify

```bash
docker compose up --build notebook
```

Open http://localhost:8989 and run notebook 01's first cell: if it prints the model name, the key made it
into the container. Then check the image from the inside, without spending any quota:

```bash
docker compose exec notebook pytest
```

Using Ollama? Inside a container, `localhost` is the container itself. Point `.env` at the laptop instead:
`LLM_BASE_URL=http://host.docker.internal:11434/v1`.


## Step 3: FastAPI, and Docker for production

So far the agent runs in a notebook, on the laptop of the person who wrote it. Here is the situation that
makes that insufficient. The marketing team wants review summaries for the 200 products in next week's
campaign, refreshed every Monday; the e-commerce team wants the "what customers say" box on each product
page to come from the same analysis; and a data scientist in another team wants to compare products across
categories. None of them should need to open a notebook, install Python, hold the LLM key or know what
an aspect table is. They need a *service*: a URL that takes a product name and returns a summary, with
one place where the key lives, the model is chosen, the quotas are managed and the logs are kept.

That is what an API is. The agent becomes a web service: anything that can send an HTTP request (a UI,
a scheduled job, a colleague's program) can ask for a summary without knowing how the agent works, and
the Step 4 UI is just the first such client.

### 3.1 The loop moves into the package

A service can import a module but not a notebook, so the loop from notebook 03 now lives in
`src/aueb_agents_ca/agent.py`: `run_agent` and `SYSTEM_PROMPT`, the same code, with `logging` instead of
prints. `tests/test_agent.py` drives it with a scripted fake model and checks that tools run and that
`max_steps` stops it.

### 3.2 The web framework

```bash
uv add --optional api fastapi "uvicorn[standard]"
```

FastAPI and uvicorn go into an *optional* group called `api`, listed under `[project.optional-dependencies]`
in `pyproject.toml`: the notebook image does not need a web server, and the API image does not need
Jupyter. Note the difference between the two kinds of optional dependency: the `dev` *group* (Jupyter,
pytest, black) is installed by default and must be excluded explicitly (`--no-dev`, as the production
image does); an *extra* such as `api` is off by default and must be asked for. So a plain `uv sync` gives
main + dev but no FastAPI, which is why the quick start says `uv sync --extra api`; without it,
`tests/test_api.py` and `uvicorn` cannot import FastAPI.

### 3.3 The service: `src/aueb_agents_ca/api.py`

Read it top to bottom; it is short.

| Part | What it does |
|---|---|
| `lifespan` | runs once at startup: reads the settings and creates the LLM client, the HTTP client and the search engine that every request shares; closes them at shutdown |
| `get_tools` | builds a fresh `ReviewTools` per request from those shared clients. Declared as a dependency with `Depends`, FastAPI's own dependency injection, so a test can swap it for a fake |
| `SummariseRequest`, `SummariseResponse` | Pydantic models: the contract of the endpoint. FastAPI validates input against them and documents them at `/docs` |
| `GET /health` | says the service is up and which model it uses; makes no model call |
| `POST /summarise` | runs the agent for one product and returns the summary and the number of model calls |
| `POST /ask` | runs the agent on a free-form task (what the chat page sends); off-topic tasks are declined by the system prompt |
| `POST /summarise/stream` | the same as `/summarise`, but the response is *streamed*: the agent's log lines as they happen, one JSON object per line, and the result as the last line. See 4.4 |

`tests/test_api.py` exercises both endpoints through FastAPI's test client with the model and the tools
replaced by fakes, and checks that bad input is rejected with `422`:

```bash
uv run pytest
```

Run it outside Docker with:

```bash
uv run uvicorn aueb_agents_ca.api:app --reload
```

### 3.4 The production image: `setup/Dockerfile`

Same skeleton as `setup/Dockerfile_dev`, different choices. Put the two files side by side:

| | development (`Dockerfile_dev`) | production (`Dockerfile`) |
|---|---|---|
| dependencies | everything, including Jupyter, pytest, black | runtime only: `uv sync --frozen --no-dev --extra api` |
| code | mounted from the laptop, edits are live | copied into the image: it is complete on its own |
| user | root (needed for Jupyter in a container) | `appuser`, created in the image: a service never runs as root |
| process | Jupyter Lab | `uvicorn`, one process, no `--reload` |
| health | none | `HEALTHCHECK` on `/health`, so Docker can tell a hung service from a healthy one |

### 3.5 Development on the production image: `docker-compose.yml`

The Dockerfile is the production definition. Compose is the local run configuration, and it adds two
things on top of the same image for development: a mounted `src/` and `--reload`, so the server restarts
when you edit a file. Nothing in compose is used in production, where the image runs exactly as built.

```yaml
  api:
    build:
      context: .
      dockerfile: setup/Dockerfile
    command: uvicorn aueb_agents_ca.api:app --host 0.0.0.0 --port 8000 --reload   # dev only
    ports:
      - "8000:8000"
    env_file: .env
    volumes:
      - ./src:/app/src   # dev only: the code that --reload watches
```

### 3.6 Logs: knowing what the agent did

A service you cannot watch needs to tell you what it did. `src/aueb_agents_ca/log.py` sets up one format
for every line, printed to stdout, which is where Docker collects logs:

```
2026-11-03 21:40:12,118 INFO [3f9a1c2e] aueb_agents_ca.agent: step 1: model asks for analyze_pages({"urls": [...]})
```

The bracketed value is the **request id**: the API gives each request one, every log line produced while
serving it carries it, and the response returns it in the `X-Request-ID` header. When two requests
overlap, their lines can still be told apart; when a user reports a bad answer, the id finds its trace.

What gets logged, at `INFO`: startup (model and provider), each request (product, duration, number of
model calls), each model call (duration, finish reason, prompt and completion tokens), each tool call the
model asks for (name and arguments) and what it returned (size and duration), and inside the tools each
search (query and result count), each fetch (URL and size), each extraction (opinions found) and each
aspect merge. `LOG_LEVEL=DEBUG` in `.env` adds the full tool results and extracted opinions.

```bash
docker compose logs -f api
```

`LOG_LEVEL` and `LOG_DIR` are two more lines in `.env.example`, read by `settings.py` like the others.

The same lines also go to a file, `logs/app.log`, mounted from the container so it lands in the repo
(`./logs:/app/logs` in both compose services; `logs/*` is git-ignored and `logs/` is in `.dockerignore`).
The file *rotates*: at 5 MB it becomes `app.log.1` and a fresh `app.log` starts, and the oldest of the
five files is deleted, so the folder can never fill a disk. `LOG_DIR` in `.env` moves it; empty means
stdout only. In a real deployment you would usually keep only stdout and let the platform collect it,
but a local file is the quickest way to read what happened after the fact.

### 3.7 Run and verify

```bash
docker compose up --build api
```

Then, in cost order:

1. `curl localhost:8000/health` and open http://localhost:8000/docs. No model call.
2. One real request, about twenty model calls on the free tier, so run it once:

```bash
curl -s -X POST localhost:8000/summarise -H 'Content-Type: application/json' -d '{"product": "Sony WH-1000XM5"}' | jq -r .summary
```

The response is JSON and `summary` is Markdown; `jq -r` unescapes it so it reads as written. Rendering it is
the job of a client, which is what Step 4 builds. The `/docs` page can send the same request from a form.

## Step 4: a simple UI

The API answers programs. People need pages: a dashboard that says whether the service is up, a form
that turns a product name into a rendered summary, and a chat box for questions in their own words.
The pages are *clients* of the API, and that is the division of labour to notice: the API knows how to
run the agent; the pages only know how to ask and display.

### 4.1 The pages: `src/aueb_agents_ca/ui/`

```bash
uv add --optional ui streamlit
```

Streamlit turns a script into a web page: every widget call (`st.text_input`, `st.button`,
`st.chat_input`) draws something, and the script re-runs from the top on every interaction. A folder with
an entry script and a `pages/` directory becomes a multi-page app with a sidebar, pages ordered by the
number in their file name.

| File | What it does |
|---|---|
| `client.py` | the only place the UI talks to the API: `api_health`, `summarise`, `ask`. Each returns the body and the `X-Request-ID` header. `API_URL` is the one setting: `http://localhost:8000` on a laptop, `http://api:8000` inside compose |
| `Home.py` | the health dashboard: status, model and latency as metrics, then the embedding model, the tools the agent can call and a link to `/docs`. If the API is down it says so and how to start it. Costs no model call |
| `pages/1_📝_Summarise.py` | a list of example products to pick from, the product box, and the equivalent `curl` command to copy. A slider sets `max_steps`, the loop's safety net (1 to 20, default 8). The button calls `/summarise/stream`: while the agent runs, its log lines appear in a status box; when the result arrives the box folds away and the summary is rendered, with the model calls and the request id; the raw JSON sits in an expander |
| `pages/2_💬_Chat.py` | a chat box that posts each message to `/ask`. The history lives in `st.session_state`; suggested questions and the `max_steps` slider sit in the sidebar |

The chat page is honest about what it is: each message is an independent run of the agent, which
searches, analyses, answers and forgets. There is no memory between messages, so the product must be
named every time, and a question that is not about product reviews gets the "not supported" answer
from the system prompt. Giving the agent memory of the conversation would be the next lesson.

`/ask` is the one endpoint added to the API for the chat: it takes a free-form `task` instead of a
product name and runs the same loop. `/summarise` is now a thin wrapper that builds the task from the
product.

Run the pages locally, with the API already running:

```bash
uv run streamlit run src/aueb_agents_ca/ui/Home.py
```

`tests/test_ui.py` tests the API calls against a fake API (an `httpx` mock transport) and renders every
page with Streamlit's `AppTest` against an address where nothing listens, checking that each shows the
"not reachable" message.

### 4.2 A second service on the production image

The UI needs Streamlit, so the production image installs both extras (`--extra api --extra ui`), and
compose runs the same image twice with different commands:

```yaml
  ui:
    build:
      context: .
      dockerfile: setup/Dockerfile     # the same image as the api; only the command differs
    command: streamlit run src/aueb_agents_ca/ui/Home.py --server.port 8501 --server.address 0.0.0.0
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000        # containers reach each other by service name, not localhost
    depends_on:
      - api
```

Two things to notice. `API_URL` points at `api`, not `localhost`: compose gives every service a DNS name
on a private network, and `localhost` inside the UI container is the UI container. And the `ui` service
has its own health check, on Streamlit's `/_stcore/health`, because the one baked into the image checks
the API's port.

### 4.3 Watching the agent work: streaming

An agent run takes about a minute. A spinner for a minute is poor manners, and the API already knows
what is happening: it logs every search, fetch, extraction and model call. `POST /summarise/stream`
sends those lines to the client *as they happen*, and the result when it is ready. The format is
newline-delimited JSON: one object per line, `{"log": "..."}` while the agent runs, then
`{"result": {...}}` or `{"error": "..."}`. Try it in a terminal and watch the lines arrive:

```bash
curl -N -X POST localhost:8000/summarise/stream -H 'Content-Type: application/json' -d '{"product": "Sony WH-1000XM5"}'
```

How it works, in `api.py`: the request id that tags every log line is what makes it possible. The endpoint
attaches a handler that collects the lines carrying *this* request's id into a queue, runs the agent in
a thread (with the request id copied into it), and yields from the queue while the thread works. The
Summarise page reads the stream with `httpx`, writes the lines into a Streamlit status box, and collapses
the box when the result line arrives, so the logs stay one click away under the summary.

### 4.4 Run and verify

```bash
docker compose up --build api ui
```

Open http://localhost:8501. The home page's metrics prove the UI reached the API across the compose
network, with no model call made. On the Summarise page, pick an example and press the button: about
twenty model calls, and the summary renders. The request id in the caption matches the `[id]` in
`docker compose logs api`, which is how you find the trace of one run.
