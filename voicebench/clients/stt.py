"""Realtime STT WebSocket wrapper (design.md section 6).

Raw protocol, mirroring what a developer writes today:
  URL: wss://api.together.ai/v1/realtime?model={stt}&input_audio_format=pcm_s16le_16000
  Headers: Authorization: Bearer <key>, OpenAI-Beta: realtime=v1
  VAD on connect: turn_detection=server_vad&threshold=0.3&min_silence_duration_ms=500
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Awaitable, Callable

import websockets

from . import ws_connect
from ..clock import b64, mono_ns

REALTIME_URL = "wss://api.together.ai/v1/realtime"

EventCallback = Callable[[dict, int], None]


class STTSession:
    def __init__(self, api_key: str, model: str, *,
                 vad: bool = True, silence_ms: int = 500, threshold: float = 0.3,
                 sample_rate: int = 16000):
        self.api_key = api_key
        self.model = model
        self.vad = vad
        self.silence_ms = silence_ms
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.ws: websockets.WebSocketClientProtocol | None = None
        self._recv_task: asyncio.Task | None = None
        self._listeners: list[EventCallback] = []
        self.t_created: int | None = None
        self.closed = False

    @property
    def url(self) -> str:
        vad_param = (
            f"&turn_detection=server_vad&threshold={self.threshold}"
            f"&min_silence_duration_ms={self.silence_ms}"
            if self.vad else "&turn_detection=none"
        )
        return (f"{REALTIME_URL}?model={self.model}"
                f"&input_audio_format=pcm_s16le_{self.sample_rate}{vad_param}")

    async def connect(self, timeout: float = 20.0) -> int:
        """Open the socket and wait for session.created. Returns connect time ms."""
        t0 = mono_ns()
        self.ws = await ws_connect(
            self.url,
            {
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1",
            },
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
        """Wait for the next event of a given type received at/before since_ns."""
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

    async def append(self, pcm: bytes):
        await self._send({"type": "input_audio_buffer.append", "audio": b64(pcm)})

    async def commit(self):
        await self._send({"type": "input_audio_buffer.commit"})

    async def update_vad(self, *, silence_ms: int | None = None,
                         threshold: float | None = None):
        """transcription_session.updated: change VAD mid-session."""
        if silence_ms is not None:
            self.silence_ms = silence_ms
        if threshold is not None:
            self.threshold = threshold
        await self._send({
            "type": "transcription_session.updated",
            "session": {
                "turn_detection": "server_vad",
                "threshold": self.threshold,
                "min_silence_duration_ms": self.silence_ms,
            },
        })

    async def _send(self, msg: dict):
        if self.ws is None:
            raise RuntimeError("STTSession not connected")
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
