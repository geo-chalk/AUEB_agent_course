"""The agent loop: send the conversation and the tool schemas, run the tools the model asks for, repeat."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageParam

from aueb_agents_ca.tools import TOOL_SCHEMAS, ReviewTools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a customer-analytics assistant. Given a product, find its reviews online and write a
short summary for a marketing team: overall sentiment, the aspects people praise, the aspects they complain
about, each backed by a quote, and the list of sources.
Plan: run search_reviews two or three times with different wording (for example the product name alone,
'long term review', and 'vs' a rival) to collect as many distinct review pages as possible; then call
analyze_pages once with every review page found, skipping shops, adverts and duplicates; then write the
summary from the returned table. Never invent reviews or quotes.
If the task is not about summarising the reviews of a product, reply that it is not supported and stop."""

# The conversation mixes two kinds of entries: the dicts we write (system, user, tool messages) and the
# message objects the model returns, which we append as they are.
Conversation = list[ChatCompletionMessageParam | ChatCompletionMessage]


def run_agent(
    *, task: str, llm: OpenAI, tools: ReviewTools, model: str, max_steps: int = 8
) -> tuple[str, Conversation]:
    """Run the agent loop on one task. Returns the final answer and the full conversation."""
    messages: Conversation = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task}]
    logger.info("agent start: task=%r model=%s max_steps=%d", task, model, max_steps)
    started = time.perf_counter()

    for step in range(max_steps):
        t0 = time.perf_counter()
        response = llm.chat.completions.create(model=model, messages=messages, tools=TOOL_SCHEMAS)
        message = response.choices[0].message
        messages.append(message)  # the object itself: some providers attach fields that must be sent back
        usage = getattr(response, "usage", None)
        logger.info(
            "step %d: model call took %.1fs, finish_reason=%s, prompt_tokens=%s, completion_tokens=%s",
            step,
            time.perf_counter() - t0,
            response.choices[0].finish_reason,
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
        )

        if not message.tool_calls:  # no tool requested: this is the final answer
            logger.info(
                "agent done: %d model calls, %d characters, %.1fs",
                step + 1,
                len(message.content or ""),
                time.perf_counter() - started,
            )
            return message.content, messages

        for call in message.tool_calls:  # run every requested tool and report back
            args: dict[str, Any] = json.loads(call.function.arguments)
            logger.info("step %d: model asks for %s(%s)", step, call.function.name, json.dumps(args)[:200])
            t0 = time.perf_counter()
            result: Any = tools.registry[call.function.name](**args)
            content = json.dumps(result)
            logger.info(
                "step %d: %s returned %d characters in %.1fs",
                step,
                call.function.name,
                len(content),
                time.perf_counter() - t0,
            )
            logger.debug("step %d: %s result: %s", step, call.function.name, content)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})

    logger.warning("agent stopped: %d steps without a final answer", max_steps)
    return "Stopped: too many steps without a final answer.", messages
