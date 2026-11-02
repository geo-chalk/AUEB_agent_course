# Customer Analytics: a simple review-summarizer agent

Course material for the AUEB MSc Data Science, *Advanced Customer Analytics*.

We build a small **agent** that, given a product name, searches the web for reviews,
analyses them (which aspects people discuss, with what sentiment, backed by quotes) and writes a
short, grounded summary with an LLM.
Along the way we learn four engineering tools:

| Step | Topic | Git tag |
|------|-------|---------|
| 1 | uv, repo setup, Python packages, notebooks | `step-1-uv` |
| 2 | Docker | `step-2-docker` |
| 3 | FastAPI | `step-3-api` |
| 4 | A simple UI (Streamlit) | `step-4-ui` |

Everything runs for free: the LLM is called through the Gemini free tier (or Groq, or a local Ollama),
and web search uses DuckDuckGo (no key).

To see the code at the end of any step: `git checkout step-1-uv` (etc.).

## Project structure

```
aueb_agents_CA/
├── README.md
├── pyproject.toml          # project metadata and dependencies (managed by uv)
├── uv.lock                 # exact versions, so everyone installs the same thing
├── .python-version         # Python version uv will use (3.13)
├── .env.example            # template for your secrets; copy to .env
├── notebooks/              # Step 1 lessons live here (img/ holds their figures)
├── src/aueb_agents_ca/     # the Python package: settings.py (config), tools.py (the agent's tools)
└── tests/                  # pytest tests that run without internet or API keys
```

## Quick start (clone and run)

```bash
git clone <this repo> && cd aueb_agents_CA
uv sync                      # creates .venv and installs everything from uv.lock
cp .env.example .env         # then put your Gemini key in .env
uv run jupyter lab           # open notebooks/
```

[Install uv](https://docs.astral.sh/uv/getting-started/installation/) first if you do not have it.

## Choosing an LLM provider

The code talks to any provider that implements the OpenAI chat-completions API. Pick one in `.env`:

| Provider | Key | Free limits | Notes |
|----------|-----|-------------|-------|
| **Gemini** (default) | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | `gemini-3.5-flash-lite`: 15 req/min, 250K tokens/min, 500 req/day. `gemini-embedding-001`: 100 req/min, 1K req/day, and every text in a batch counts as a request. `gemini-3.8-flash`: 5 req/min, 20 req/day | One agent run costs about 20 chat requests and up to 40 embedding requests, so a key is good for roughly 20 runs a day. Check your own numbers in AI Studio under Rate limits. Free-tier prompts may be used by Google to improve its products |
| Groq | [console.groq.com](https://console.groq.com) | 30 req/min, 8K tokens/min, 1K req/day | very fast; the agent loop can hit the tokens/min limit |
| Ollama | none | unlimited | runs on your laptop; needs 8 GB RAM; smaller models |

---

# Recreate the repo from scratch

Each step below lists the exact commands that produced it.

## Step 1: uv, repo setup, notebooks

### 1.1 Create the project

```bash
mkdir aueb_agents_CA && cd aueb_agents_CA
git init -b main
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
| `notebooks/01_llm_call.ipynb` | one chat completion; switch provider by changing `.env` |
| `notebooks/02_tools.ipynb` | the tools: search, fetch page, and an aspect-based analysis (model extracts opinions, code counts them) |
| `notebooks/03_agent_loop.ipynb` | the agent loop: model asks for tools, we run them, repeat |

### 1.6 The package and its tests

The tools used by notebook 03 live in the package, `src/aueb_agents_ca/`, so that the API can reuse them
later: `settings.py` reads the configuration, `tools.py` holds the tools. They receive their dependencies
(LLM client, HTTP client, search engine, settings) from the caller, which is what lets `tests/test_tools.py`
test them with fakes, offline. Copy the three files from this tag, then run:

```bash
uv run pytest
```

The package version of the tools is the notebook version plus the hardening a real service needs:
search failures returned as text, embedding requests batched and capped, empty pages skipped, logging
instead of prints.

### 1.7 Commit and tag

```bash
git add . && git commit -m "Step 1: uv setup, notebooks and tools package"
git tag step-1-uv
```

## Step 2: Docker

*Coming soon.*

## Step 3: FastAPI

*Coming soon.*

## Step 4: UI

*Coming soon.*
