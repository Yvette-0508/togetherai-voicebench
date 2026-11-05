"""Streamed chat with sentence splitting and a tool loop (design.md 7.3).

- Streams the chat completion; records time of first token and of the first
  complete sentence. A sentence ends at (?<=[.!?])\s or at end of stream.
- Each completed sentence goes to TTS as soon as it is available (the pattern
  in Together's phone-agent guide).
- Tool loop: on tool_calls, run a local stub (get_weather returns fixed JSON),
  validate arguments against the schema, append the result, stream a second call.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from openai import AsyncOpenAI

from ..clock import mono_ns

SENTENCE_END = re.compile(r"(?<=[.!?])\s")


@dataclass
class LLMResult:
    text: str = ""
    t_first_token: int | None = None
    first_sentence: str = ""
    t_first_sentence: int | None = None   # when the first sentence completed
    sentences: list[tuple[str, int]] = field(default_factory=list)  # (text, t_ns)
    tool_calls: list[dict] = field(default_factory=list)  # validated calls
    tool_valid: bool | None = None        # None = no tools were used
    n_tokens: int = 0


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["c", "f"]},
            },
            "required": ["city"],
        },
    },
}


def _validate_tool_args(name: str, args: Any) -> bool:
    """Validate get_weather arguments against the schema."""
    if name != "get_weather" or not isinstance(args, dict):
        return False
    if not isinstance(args.get("city"), str) or not args["city"]:
        return False
    unit = args.get("unit")
    return unit is None or unit in ("c", "f")


def get_weather(city: str, unit: str = "f") -> dict:
    """Local stub: returns a fixed JSON immediately."""
    return {"city": city, "temperature": 68, "unit": unit, "conditions": "clear"}


class SentenceSplitter:
    """Incrementally split a token stream into sentences."""

    def __init__(self, on_sentence: Callable[[str], None] | None = None):
        self.buf = ""
        self.sentences: list[str] = []
        self.on_sentence = on_sentence

    def feed(self, text: str) -> list[str]:
        self.buf += text
        out = []
        while (m := SENTENCE_END.search(self.buf)):
            sent, self.buf = self.buf[:m.end()].strip(), self.buf[m.end():]
            if sent:
                out.append(sent)
        self._emit(out)
        return out

    def flush(self) -> list[str]:
        out = [self.buf.strip()] if self.buf.strip() else []
        self.buf = ""
        self._emit(out)
        return out

    def _emit(self, new: list[str]):
        self.sentences.extend(new)
        if self.on_sentence:
            for s in new:
                self.on_sentence(s)


async def stream_chat(client: AsyncOpenAI, model: str, system: str, user: str,
                      *, max_tokens: int = 120, temperature: float = 0.3,
                      tools: list[dict] | None = None,
                      on_sentence: Callable[[str], Awaitable[None]] | None = None,
                      on_token: Callable[[str, float], None] | None = None,
                      on_event: Callable[[str, dict], None] | None = None) -> LLMResult:
    """Run one streamed turn (with optional tool call + second stream).

    on_sentence: awaited with each completed sentence as soon as it forms.
    on_token: called with (token_text, offset_ms_from_call_start) — used by T4.
    on_event: called with (name, meta) for coarse events (first_token etc.).
    """
    result = LLMResult()
    t0 = mono_ns()
    splitter = SentenceSplitter()

    async def emit_sentence(sent: str):
        t = mono_ns()
        result.sentences.append((sent, t))
        if result.t_first_sentence is None:
            result.first_sentence = sent
            result.t_first_sentence = t
        if on_sentence:
            await on_sentence(sent)

    async def _stream(messages, use_tools):
        stream = await client.chat.completions.create(
            model=model, messages=messages, stream=True,
            max_tokens=max_tokens, temperature=temperature,
            tools=use_tools or None,
        )
        tool_acc: dict[int, dict] = {}
        async for chunk in stream:
            now = mono_ns()
            if result.t_first_token is None:
                result.t_first_token = now
                if on_event:
                    on_event("first_token", {"ms": (now - t0) / 1e6})
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                result.text += delta.content
                result.n_tokens += 1
                if on_token:
                    on_token(delta.content, (now - t0) / 1e6)
                for sent in splitter.feed(delta.content):
                    await emit_sentence(sent)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    acc = tool_acc.setdefault(
                        tc.index, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        acc["id"] = tc.id
                    if tc.function and tc.function.name:
                        acc["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        acc["arguments"] += tc.function.arguments
        return [tool_acc[i] for i in sorted(tool_acc)]

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    calls = await _stream(messages, tools)

    if calls:
        # Tool loop: validate args, run the local stub, stream the second call.
        all_valid = True
        messages.append({
            "role": "assistant",
            "content": result.text or None,
            "tool_calls": [{
                "id": c["id"], "type": "function",
                "function": {"name": c["name"], "arguments": c["arguments"]},
            } for c in calls],
        })
        for c in calls:
            try:
                args = json.loads(c["arguments"]) if c["arguments"] else {}
            except json.JSONDecodeError:
                args, valid = {}, False
            else:
                valid = _validate_tool_args(c["name"], args)
            all_valid = all_valid and valid
            outcome = get_weather(args.get("city", ""), args.get("unit", "f"))
            result.tool_calls.append({"name": c["name"], "arguments": args, "valid": valid})
            messages.append({
                "role": "tool", "tool_call_id": c["id"],
                "content": json.dumps(outcome),
            })
        result.tool_valid = all_valid
        result.text = ""
        splitter = SentenceSplitter()
        await _stream(messages, None)

    for sent in splitter.flush():
        await emit_sentence(sent)
    return result
