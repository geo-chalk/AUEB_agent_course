"""The loop runs the tools the model asks for and stops when it answers, proven with a scripted model."""

import json

from fakes import FAKE_TOOLS, ScriptedLLM, message, tool_call

from aueb_agents_ca.agent import run_agent


def test_loop_runs_the_requested_tool_then_returns_the_answer():
    llm = ScriptedLLM(
        [message(tool_calls=[tool_call("search_reviews", query="XM5 review")]), message(content="done")]
    )

    answer, conversation = run_agent("Summarise XM5 reviews", llm, FAKE_TOOLS, "fake-model")

    assert answer == "done"
    assert len(llm.calls) == 2  # one call asked for the tool, one produced the answer
    roles = [m["role"] if isinstance(m, dict) else "assistant" for m in conversation]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert json.loads(conversation[3]["content"]) == [{"title": "t", "url": "u", "snippet": "s"}]
    assert conversation[3]["tool_call_id"] == "call_1"  # the result is tied to the call it answers


def test_loop_stops_after_max_steps():
    llm = ScriptedLLM([message(tool_calls=[tool_call("search_reviews", query="x")])] * 3)

    answer, _ = run_agent("task", llm, FAKE_TOOLS, "fake-model", max_steps=3)

    assert answer.startswith("Stopped")
    assert len(llm.calls) == 3
