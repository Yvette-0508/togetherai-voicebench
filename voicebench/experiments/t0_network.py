"""T0. Network baseline (design.md section 9, P0).

Separates network from server time:
- 20 small authenticated requests (list models): median round trip
- 10 cold connects each for the STT and TTS WebSockets: connect -> session.created
"""
from __future__ import annotations

import asyncio

import httpx

from ..clients.stt import STTSession
from ..clients.tts import TTSSession
from ..clock import EventLog, mono_ns

MODELS_URL = "https://api.together.ai/v1/models"


async def run(cfg: dict, resume: bool = False):
    from ..config import require_key
    require_key(cfg)
    log = EventLog("t0")
    try:
        # 1) HTTP round trips.
        async with httpx.AsyncClient(timeout=30.0) as client:
            for i in range(20):
                t0 = mono_ns()
                r = await client.get(MODELS_URL, headers={
                    "Authorization": f"Bearer {cfg['api_key']}"})
                rtt_ms = (mono_ns() - t0) / 1e6
                log.log(run_id=f"t0-rtt-{i:03d}", experiment="t0", config="rtt",
                        turn=i, direction="recv", type_="http.rtt",
                        meta={"status": r.status_code, "ms": rtt_ms})
                await asyncio.sleep(0.2)

        # 2) Cold WebSocket connects.
        for kind, session_cls, kwargs in (
            ("stt", STTSession, {}),
            ("tts", TTSSession, {}),
        ):
            for i in range(10):
                if kind == "stt":
                    s = session_cls(cfg["api_key"], cfg["models"]["stt"],
                                    sample_rate=cfg["audio"]["stt_sample_rate"])
                else:
                    s = session_cls(cfg["api_key"], cfg["models"]["tts"],
                                    cfg["models"]["voice"],
                                    sample_rate=cfg["audio"]["tts_sample_rate"])
                try:
                    ms = await s.connect(timeout=30.0)
                    log.log(run_id=f"t0-{kind}-connect-{i:03d}", experiment="t0",
                            config=f"{kind}_connect", turn=i, direction="recv",
                            type_="session.created", meta={"ms": ms})
                    print(f"  {kind} cold connect {i}: {ms} ms")
                except Exception as e:
                    log.log(run_id=f"t0-{kind}-connect-{i:03d}", experiment="t0",
                            config=f"{kind}_connect", turn=i, direction="meta",
                            type_="connect.failed", meta={"error": str(e)})
                finally:
                    await s.close()
                await asyncio.sleep(0.3)
    finally:
        log.close()
