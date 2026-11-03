"""The tools can be tested without the internet or an API key, because their dependencies are injected."""

import json
from types import SimpleNamespace

import httpx
from ddgs.exceptions import DDGSException

from aueb_agents_ca.settings import Settings
from aueb_agents_ca.tools import ReviewTools

SETTINGS = Settings(base_url="http://fake", api_key="fake", model="fake-model", embedding_model=None)


class FakeLLM:
    """Stands in for the OpenAI client: returns a fixed list of opinions for every extraction call."""

    def __init__(self, *, opinions: list[dict]) -> None:
        content = json.dumps({"opinions": opinions})
        message = SimpleNamespace(content=content)
        response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response))


class FakeSearch:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def text(self, query: str, max_results: int) -> list[dict]:
        self.queries.append(query)
        return [{"title": "A review", "href": "https://example.com/review", "body": f"snippet for {query}"}]


def make_tools(llm=None, http=None, search=None) -> ReviewTools:
    return ReviewTools(
        llm=llm or FakeLLM(opinions=[]),
        settings=SETTINGS,
        http=http
        or httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, text="<p>hi</p>"))),
        search=search or FakeSearch(),
    )


def test_search_keeps_only_title_url_snippet():
    results = make_tools().search_reviews(query="Sony WH-1000XM5")
    assert results == [
        {
            "title": "A review",
            "url": "https://example.com/review",
            "snippet": "snippet for Sony WH-1000XM5",
        }
    ]


def test_fetch_page_reports_http_errors_as_text():
    http = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(403)))
    text = make_tools(http=http).fetch_page(url="https://blocked.example.com")
    assert text.startswith("ERROR: could not fetch https://blocked.example.com")


def test_analyze_reviews_counts_per_aspect_and_sorts_by_volume():
    llm = FakeLLM(
        opinions=[
            {"aspect": "Price", "sentiment": "negative", "quote": "too expensive"},
            {"aspect": "comfort", "sentiment": "positive", "quote": "very comfortable"},
            {"aspect": "comfort", "sentiment": "positive", "quote": "soft pads"},
            {"aspect": "comfort", "sentiment": "mixed", "quote": "warm after an hour"},
        ]
    )
    table = make_tools(llm=llm).analyze_reviews(texts=["one review"])
    assert [row["aspect"] for row in table] == ["comfort", "price"]  # most discussed first, names lower-cased
    assert table[0]["positive"] == 2 and table[0]["mixed"] == 1
    assert table[0]["quotes"] == ["very comfortable", "soft pads"]  # capped at two quotes


def test_analyze_pages_skips_failed_pages_but_reports_them():
    def handler(request):
        return (
            httpx.Response(403)
            if "blocked" in str(request.url)
            else httpx.Response(200, text="<p>Great sound.</p>")
        )

    tools = make_tools(
        llm=FakeLLM(opinions=[{"aspect": "sound", "sentiment": "positive", "quote": "Great sound."}]),
        http=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    tools.pages["https://ok.example.com"] = "Great sound."  # already fetched earlier: reused, not downloaded
    result = tools.analyze_pages(urls=["https://ok.example.com", "https://blocked.example.com"])
    assert result["pages_analyzed"] == ["https://ok.example.com"]
    assert result["pages_failed"] == ["https://blocked.example.com"]
    assert result["aspects"][0]["aspect"] == "sound"


def test_settings_from_env_accepts_a_path(tmp_path, monkeypatch):
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "EMBEDDING_MODEL"):
        monkeypatch.delenv(name, raising=False)
    env_file = tmp_path / "custom.env"
    env_file.write_text(
        "LLM_BASE_URL=http://fake\nLLM_API_KEY=fake\nLLM_MODEL=fake-model\nEMBEDDING_MODEL=\n"
    )

    settings = Settings.from_env(dotenv_path=env_file)

    assert settings.model == "fake-model"
    assert settings.embedding_model is None  # empty value means "skip the merge"


def test_search_reports_backend_failures_as_text():
    class FailingSearch:
        def text(self, query: str, max_results: int) -> list[dict]:
            raise DDGSException("No results found.")

    result = make_tools(search=FailingSearch()).search_reviews(query="Sony WH-1000XM5")
    assert isinstance(result, str) and result.startswith("ERROR: search failed")


def test_embed_sends_inputs_in_batches_of_at_most_100():
    calls: list[int] = []

    def create(model: str, input: list[str]):
        calls.append(len(input))
        data = [SimpleNamespace(embedding=[1.0, float(i)]) for i, _ in enumerate(input)]
        return SimpleNamespace(data=data)

    llm = FakeLLM(opinions=[])
    llm.embeddings = SimpleNamespace(create=create)
    settings = Settings(
        base_url="http://fake", api_key="fake", model="fake-model", embedding_model="fake-embed"
    )
    tools = ReviewTools(llm=llm, settings=settings, http=make_tools().http, search=FakeSearch())

    vectors = tools.embed(texts=[f"aspect {i}" for i in range(250)])

    assert calls == [100, 100, 50]
    assert vectors.shape == (250, 2)


def test_merge_aspects_embeds_only_the_most_frequent_names():
    embedded: list[list[str]] = []

    def create(model: str, input: list[str]):
        embedded.append(list(input))
        # orthogonal unit vectors: nothing is similar to anything, so nothing merges
        data = [
            SimpleNamespace(embedding=[1.0 if j == i else 0.0 for j in range(len(input))])
            for i in range(len(input))
        ]
        return SimpleNamespace(data=data)

    llm = FakeLLM(opinions=[])
    llm.embeddings = SimpleNamespace(create=create)
    settings = Settings(
        base_url="http://fake", api_key="fake", model="fake-model", embedding_model="fake-embed"
    )
    tools = ReviewTools(llm=llm, settings=settings, http=make_tools().http, search=FakeSearch())

    names = ["comfort"] * 5 + ["price"] * 3 + [f"rare aspect {i}" for i in range(10)]
    mapping = tools.merge_aspects(names=names, max_names=4)

    assert embedded == [["comfort", "price", "rare aspect 0", "rare aspect 1"]]  # top 4 by frequency only
    assert mapping["rare aspect 9"] == "rare aspect 9"  # the tail is kept, unmerged
    assert len(mapping) == 12


def test_search_runs_the_query_exactly_as_given():
    search = FakeSearch()
    make_tools(search=search).search_reviews(query="Sony WH-1000XM5 long term review")
    assert search.queries == ["Sony WH-1000XM5 long term review"]  # no silent rewriting


def test_analyze_pages_treats_empty_pages_as_failed():
    http = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text="<html></html>"))
    )
    result = make_tools(http=http).analyze_pages(urls=["https://empty.example.com"])
    assert result == {"error": "none of the pages could be fetched", "failed": ["https://empty.example.com"]}


def test_only_search_and_analyze_are_exposed_to_the_model():
    from aueb_agents_ca.tools import TOOL_SCHEMAS

    assert list(make_tools().registry) == ["search_reviews", "analyze_pages", "find_product_image"]
    assert [schema["function"]["name"] for schema in TOOL_SCHEMAS] == [
        "search_reviews",
        "analyze_pages",
        "find_product_image",
    ]
