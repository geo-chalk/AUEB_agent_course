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
├── src/aueb_agents_ca/     # the Python package: settings.py (config), tools.py (tools); Step 3 adds agent.py
├── tests/                  # pytest tests that run without internet or API keys
├── setup/Dockerfile_dev    # Step 2: the development image (notebooks); Step 3 adds the production one
├── docker-compose.yml      # Step 2: how to run the image (ports, .env, mounted folders)
└── .dockerignore           # what never enters the image (.env, .venv, .git)
```

## Quick start (clone and run)

```bash
git clone <this repo> && cd aueb_agents_CA
uv sync                      # creates .venv and installs everything from uv.lock
cp .env.example .env         # then put your Gemini key in .env
uv run jupyter lab           # open notebooks/
```

[Install uv](https://docs.astral.sh/uv/getting-started/installation/) first if you do not have it.

Or, with Docker installed, skip uv entirely:

```bash
docker compose up --build        # then open http://localhost:8989
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
docker compose up --build
```

Open http://localhost:8989 and run notebook 01's first cell: if it prints the model name, the key made it
into the container. Then check the image from the inside, without spending any quota:

```bash
docker compose exec notebook pytest
```

Using Ollama? Inside a container, `localhost` is the container itself. Point `.env` at the laptop instead:
`LLM_BASE_URL=http://host.docker.internal:11434/v1`.


## Step 3: FastAPI, and Docker for production

*Coming soon.* Two things happen here. The agent loop leaves notebook 03 for the package
(`src/aueb_agents_ca/agent.py`), because a service can import a module but not a notebook. And the
production image differs from the development one in a few deliberate ways: runtime dependencies only
(`uv sync --frozen --no-dev`), code copied in rather than mounted, a non-root user, one process started
by `CMD` (`uvicorn`, no `--reload`), and a health check.

## Step 4: UI

*Coming soon.*
