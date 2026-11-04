"""Stand-ins for the model, shared by the agent and API tests. No network, no key."""

import json
from types import SimpleNamespace


def tool_call(*, name: str, **args) -> SimpleNamespace:
    """What the model sends when it wants a tool run: an id, the tool name and JSON arguments."""
    return SimpleNamespace(id="call_1", function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def message(*, content: str | None = None, tool_calls: list | None = None) -> SimpleNamespace:
    """A model message: either text (the final answer) or tool calls."""
    return SimpleNamespace(content=content, tool_calls=tool_calls)


class ScriptedLLM:
    """Plays back a script of messages, one per call, so a test decides what the 'model' does."""

    def __init__(self, *, script: list[SimpleNamespace]) -> None:
        self.calls: list[dict] = []
        responses = [
            SimpleNamespace(choices=[SimpleNamespace(message=m, finish_reason="stop")]) for m in script
        ]
        self._responses = iter(responses)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
        return next(self._responses)


# A tools object with just what the loop needs: a registry of callables.
FAKE_TOOLS = SimpleNamespace(
    registry={"search_reviews": lambda *, query, max_results=10: [{"title": "t", "url": "u", "snippet": "s"}]}
)
