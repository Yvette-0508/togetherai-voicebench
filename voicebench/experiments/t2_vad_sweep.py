"""T2. VAD silence sweep (design.md section 9, P0). Reuses T1's run_turn.

For each min_silence_duration_ms in the sweep, run N turns from the "paused"
corpus (two clauses joined by 300-600 ms of silence).
- cut: a `completed` event arrives before t_eos => the agent would have
  answered half a question.
- latency: t_first_audio - t_eos for turns that were not cut off.
"""
from __future__ import annotations

from openai import AsyncOpenAI

from ..clock import EventLog, existing_run_ids
from .t1_turn_latency import BASE_URL, run_config


async def run(cfg: dict, resume: bool = False):
    from ..config import require_key
    require_key(cfg)
    log = EventLog("t2")
    try:
        llm_client = AsyncOpenAI(api_key=cfg["api_key"], base_url=BASE_URL)
        resume_ids = existing_run_ids("t2") if resume else set()
        n = cfg["runs"]["t2_turns"]
        for ms in cfg["vad"]["sweep_ms"]:
            spec = {"vad": True, "silence_ms": ms, "set": "paused", "tools": False}
            await run_config(cfg, llm_client, f"silence-{ms}", spec, log, n,
                             resume_ids, experiment="t2")
    finally:
        log.close()
