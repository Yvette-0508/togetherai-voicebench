"""Realtime TTS WebSocket wrapper + HTTP TTS helpers (design.md section 6).

WebSocket (no SDK):
  wss://api.together.ai/v1/audio/speech/websocket?model={tts}&voice={voice}
      &response_format=pcm&sample_rate=24000
  client->server: input_text_buffer.append / .commit / .clear, context.cancel,
                  tts_session.updated
  server->client: session.created, conversation.item.input_text.received,
                  conversation.item.audio_output.delta (base64 PCM),
                  conversation.item.audio_output.done, context.cancelled,
                  conversation.item.tts.failed
HTTP:
  POST /v1/audio/speech (full file), and a streaming variant for raw PCM bytes.
"""
from __future__ import annotations

import asyncio
import json
from typing import Callable

import httpx
import websockets

from . import ws_connect
from ..clock import mono_ns

TTS_WS_URL = "wss://api.together.ai/v1/audio/speech/websocket"
TTS_HTTP_URL = "https://api.together.ai/v1/audio/speech"

EventCallback = Callable[[dict, int], None]


class TTSSession:
    def __init__(self, api_key: str, model: str, voice: str, *,
                 sample_rate: int = 24000, segment: str | None = None,
                 max_partial_length: int | None = None):
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.sample_rate = sample_rate
        self.segment = segment  # sentence | immediate | never | None (server default)
        self.max_partial_length = max_partial_length
        self.ws: websockets.WebSocketClientProtocol | None = None
        self._recv_task: asyncio.Task | None = None
        self._listeners: list[EventCallback] = []
        self.t_created: int | None = None
        self.closed = False

    @property
    def url(self) -> str:
        url = (f"{TTS_WS_URL}?model={self.model}&voice={self.voice}"
               f"&response_format=pcm&sample_rate={self.sample_rate}")
        if self.segment:
            url += f"&segment={self.segment}"
        if self.max_partial_length is not None:
            url += f"&max_partial_length={self.max_partial_length}"
        return url

    async def connect(self, timeout: float = 20.0) -> int:
        t0 = mono_ns()
        self.ws = await ws_connect(
            self.url,
            {"Authorization": f"Bearer {self.api_key}"},
            max_size=None,
        )
        self.closed = False
        created = asyncio.get_event_loop().create_future()

        def _on_created(ev, t_ns):
            if ev.get("type") == "session.created" and not created.done():
                created.set_result(t_ns)

        self.on(_on_created)
        self._recv_task = asyncio.create_task(self._recv_loop())
        await asyncio.wait_for(created, timeout)
        self.t_created = created.result()
        return int((self.t_created - t0) / 1e6)

    async def _recv_loop(self):
        try:
            async for raw in self.ws:
                t_ns = mono_ns()
                try:
                    ev = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                for cb in list(self._listeners):
                    try:
                        cb(ev, t_ns)
                    except Exception:
                        pass
        except (websockets.ConnectionClosed, asyncio.CancelledError):
            pass
        finally:
            self.closed = True

    def on(self, cb: EventCallback):
        self._listeners.append(cb)

    def off(self, cb: EventCallback):
        if cb in self._listeners:
            self._listeners.remove(cb)

    async def wait_for(self, type_: str, *, since_ns: int = 0,
                       predicate: Callable[[dict, int], bool] | None = None,
                       timeout: float = 10.0) -> tuple[dict, int]:
        fut = asyncio.get_event_loop().create_future()

        def _cb(ev, t_ns):
            if fut.done() or ev.get("type") != type_ or t_ns < since_ns:
                return
            if predicate and not predicate(ev, t_ns):
                return
            fut.set_result((ev, t_ns))

        self.on(_cb)
        try:
            return await asyncio.wait_for(fut, timeout)
        finally:
            self.off(_cb)

    async def append(self, text: str, context_id: str | None = None):
        msg = {"type": "input_text_buffer.append", "text": text}
        if context_id is not None:
            msg["context_id"] = context_id
        await self._send(msg)

    async def commit(self):
        await self._send({"type": "input_text_buffer.commit"})

    async def clear(self):
        """Drops queued text; text already being processed continues."""
        await self._send({"type": "input_text_buffer.clear"})

    async def cancel(self, context_id: str = "default"):
        await self._send({"type": "context.cancel", "context_id": context_id})

    async def session_update(self, **settings):
        await self._send({"type": "tts_session.updated", "session": settings})

    async def _send(self, msg: dict):
        if self.ws is None:
            raise RuntimeError("TTSSession not connected")
        await self.ws.send(json.dumps(msg))

    async def close(self):
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except (asyncio.CancelledError, Exception):
                pass
        if self.ws:
            await self.ws.close()
        self.closed = True


async def http_tts(api_key: str, model: str, voice: str, text: str, *,
                   sample_rate: int = 24000, timeout: float = 60.0) -> bytes:
    """Full-file TTS over HTTP. Returns WAV bytes."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(
            TTS_HTTP_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model, "voice": voice, "input": text,
                "response_format": "wav", "sample_rate": sample_rate,
            },
        )
        r.raise_for_status()
        return r.content


async def http_tts_stream(api_key: str, model: str, voice: str, text: str, *,
                          sample_rate: int = 24000):
    """Streaming variant: yields (chunk_bytes, t_ns) with raw PCM bytes."""
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST", TTS_HTTP_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model, "voice": voice, "input": text,
                "response_format": "raw", "sample_rate": sample_rate,
                "stream": True,
            },
        ) as r:
            r.raise_for_status()
            async for chunk in r.aiter_bytes():
                yield chunk, mono_ns()
