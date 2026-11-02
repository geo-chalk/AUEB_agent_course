"""Tools for the review-summariser agent.

Everything the tools depend on (the LLM client, the HTTP client, the search engine, the settings) is
passed in from outside instead of being created here. Importing this module has no side effects, the
same code runs against Gemini, Groq or Ollama, and the tools can be tested with fakes
(see tests/test_tools.py). `build_tools` wires the real dependencies for everyday use.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Callable

import httpx
import numpy as np
import trafilatura
from ddgs import DDGS
from ddgs.exceptions import DDGSException
from openai import OpenAI

from aueb_agents_ca.settings import Settings

logger = logging.getLogger(__name__)

OPINION_SCHEMA = {
    "type": "object",
    "properties": {
        "opinions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "aspect": {"type": "string"},
                    "sentiment": {"type": "string", "enum": ["positive", "negative", "mixed"]},
                    "quote": {"type": "string"},
                },
                "required": ["aspect", "sentiment", "quote"],
            },
        }
    },
    "required": ["opinions"],
}

EXTRACT_PROMPT = (
    "You extract opinions about a product from a review. For every opinion return the product aspect it "
    "is about (a short generic name such as 'noise cancelling', 'battery life', 'price', 'comfort'), the "
    "sentiment, and the exact words from the text that express it. Ignore text that is not an opinion "
    "about the product. Be exhaustive: go through the whole text section by section and list every "
    "opinion you find; a long review typically contains 15 to 40 opinions."
)


class ReviewTools:
    """The functions the agent can call, bound to their dependencies."""

    def __init__(self, llm: OpenAI, settings: Settings, http: httpx.Client, search: DDGS) -> None:
        self.llm = llm
        self.settings = settings
        self.http = http
        self.search = search
        self.pages: dict[str, str] = {}  # url -> text, so the model can refer to a page by its URL

    # ----- the two tools the agent sees -----

    def search_reviews(self, query: str, max_results: int = 10) -> list[dict] | str:
        """Web search, exactly as queried. Returns title, url and snippet per result."""
        try:
            results = self.search.text(query, max_results=max_results)
        except DDGSException as e:
            # The search backend throttles now and then. Report it as text; the model can rephrase and retry.
            return f"ERROR: search failed for {query!r}: {e}"
        # Keep only what the agent needs; every field here ends up as tokens in the prompt.
        return [{"title": r["title"], "url": r["href"], "snippet": r["body"]} for r in results]

    def fetch_page(self, url: str, max_chars: int = 20000) -> str:
        """Download a web page and return its readable text, cut to max_chars.

        Not exposed to the model: given the choice, a small model sometimes reads pages instead of
        analysing them. analyze_pages calls this internally.
        """
        try:
            response = self.http.get(url)
            response.raise_for_status()  # turns 403, 404, 500 ... into an exception
        except httpx.HTTPError as e:
            # A tool must never crash the agent: report the problem as text instead.
            return f"ERROR: could not fetch {url}: {e}"
        text = (trafilatura.extract(response.text) or "")[:max_chars]
        self.pages[url] = text
        return text

    def analyze_pages(self, urls: list[str]) -> dict:
        """Fetch review pages (or reuse already fetched ones) and return the aspect table."""
        texts, used, failed = [], [], []
        for url in urls:
            text = self.pages.get(url) or self.fetch_page(url)
            if not text or text.startswith("ERROR"):  # blocked, or nothing readable on the page
                failed.append(url)
            else:
                texts.append(text)
                used.append(url)
        if not texts:
            return {"error": "none of the pages could be fetched", "failed": failed}
        return {"pages_analyzed": used, "pages_failed": failed, "aspects": self.analyze_reviews(texts)}

    @property
    def registry(self) -> dict[str, Callable]:
        """Tool name -> function, so the agent loop can run what the model asks for."""
        return {"search_reviews": self.search_reviews, "analyze_pages": self.analyze_pages}

    # ----- building blocks of the analysis -----

    def extract_opinions(self, text: str) -> list[dict]:
        """Ask the model to list every opinion in a review as (aspect, sentiment, quote)."""
        response = self.llm.chat.completions.create(
            model=self.settings.model,
            messages=[{"role": "system", "content": EXTRACT_PROMPT}, {"role": "user", "content": text}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "opinions", "schema": OPINION_SCHEMA},
            },
        )
        return json.loads(response.choices[0].message.content)["opinions"]

    def embed(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        """One unit-length vector per text, as rows of a matrix. Sent in batches: providers cap the batch size."""
        vectors = []
        for start in range(0, len(texts), batch_size):
            response = self.llm.embeddings.create(
                model=self.settings.embedding_model, input=texts[start : start + batch_size]
            )
            vectors += [d.embedding for d in response.data]
        matrix = np.array(vectors)
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    def merge_aspects(self, names: list[str], threshold: float = 0.8, max_names: int = 40) -> dict[str, str]:
        """Map each aspect name to a canonical one; near-duplicates share the most frequent spelling.

        Only the `max_names` most frequent names are embedded and merged: the long tail of one-off names
        never reaches the top of the table, and embedding providers count every input as a request.
        """
        unique = [name for name, _ in Counter(names).most_common()]  # most frequent first
        mapping = {name: name for name in unique}  # default: every name is its own aspect
        head = unique[:max_names]
        if self.settings.embedding_model is None or not head:
            return mapping
        try:
            vectors = self.embed(head)
        except Exception as e:  # embedding endpoint unavailable? merge nothing, but do not crash
            logger.warning("embedding failed, skipping merge: %s", e)
            return mapping

        kept: list[str] = []
        for i, name in enumerate(head):
            similarities = [(float(vectors[i] @ vectors[head.index(k)]), k) for k in kept]
            best_score, best_name = max(similarities, default=(0.0, None))
            if best_score >= threshold:
                mapping[name] = best_name  # close enough to an existing aspect: merge into it
            else:
                mapping[name] = name  # new aspect
                kept.append(name)
        return mapping

    def analyze_reviews(self, texts: list[str], max_quotes: int = 2) -> list[dict]:
        """Per aspect: how many positive/negative/mixed opinions and a few quotes. Most discussed first."""
        opinions = []
        for text in texts:
            try:
                opinions += self.extract_opinions(text)
            except Exception as e:  # one bad page must not sink the whole analysis
                logger.warning("extraction failed for one text: %s", e)

        names = [o["aspect"].lower().strip() for o in opinions]
        canonical = self.merge_aspects(names)

        table: dict[str, dict] = {}
        for opinion, name in zip(opinions, names):
            aspect = canonical[name]
            row = table.setdefault(
                aspect, {"aspect": aspect, "positive": 0, "negative": 0, "mixed": 0, "quotes": []}
            )
            row[opinion["sentiment"]] += 1
            if len(row["quotes"]) < max_quotes:
                row["quotes"].append(opinion["quote"])

        return sorted(table.values(), key=lambda r: -(r["positive"] + r["negative"] + r["mixed"]))


# What the model sees: one JSON schema per tool. The description is all it has to decide when to call it.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_reviews",
            "description": (
                "Web search, run exactly as queried: phrase the query to find review pages, e.g. "
                "'<product> review'. Returns title, url and snippet per result, about ten per query at most. "
                "Run several differently worded searches to find more pages."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, e.g. '<product> review', '<product> long term review', '<product> vs <rival>'",
                    },
                    "max_results": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_pages",
            "description": (
                "Fetch review pages and return, per product aspect, how many positive, negative and mixed "
                "opinions they contain, with example quotes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Review page URLs: every relevant page found, excluding shops and adverts",
                    }
                },
                "required": ["urls"],
            },
        },
    },
]


def build_tools(settings: Settings | None = None) -> ReviewTools:
    """Wire the real dependencies together. Notebooks and services call this; tests build ReviewTools by hand."""
    settings = settings or Settings.from_env()
    llm = OpenAI(base_url=settings.base_url, api_key=settings.api_key, timeout=60, max_retries=3)
    http = httpx.Client(follow_redirects=True, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    return ReviewTools(llm=llm, settings=settings, http=http, search=DDGS())
