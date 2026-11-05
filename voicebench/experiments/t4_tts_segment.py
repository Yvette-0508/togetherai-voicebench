"""T4. TTS segment modes (design.md section 9, P1).

Record N LLM replies once, saving each token's text and relative timestamp.
For each segment mode (immediate | sentence | never), replay the identical
token streams into TTS with input_text_buffer.append at the original timing,
then commit. Replaying identical text and timing keeps the comparison fair.

Metrics per mode:
- ttfa: median time from the first token sent to the first audio delta.
- stalls: number of replies with at least one playback stall (section 7.4).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI

from ..clients.llm import stream_chat
from ..clients.tts import TTSSession
from ..clock import RESULTS, EventLog, existing_run_ids, mono_ns, unb64
from ..corpus import load_manifest
from ..playback import PlaybackSimulator
from .t1_turn_latency import BASE_URL

REPLIES_PATH = RESULTS / "raw" / "t4_replies.json"
MODES = ("immediate", "sentence", "never")


async def record_replies(cfg: dict, llm_client: AsyncOpenAI, n: int) -> list[dict]:
    """Stream n LLM replies; keep every token's text and offset from call start."""
    manifest = load_manifest()
    prompts = [e["text"] for e in manifest["sets"]["short"]]
    replies = []
    for i in range(n):
        tokens: list[list] = []

        def on_token(text, offset_ms):
            tokens.append([text, offset_ms])

        try:
            result = await stream_chat(
                llm_client, cfg["models"]["llm"], cfg["llm"]["system"],
                prompts[i % len(prompts)],
                max_tokens=cfg["llm"]["max_tokens"],
                temperature=cfg["llm"]["temperature"],
                extra_body=cfg["llm"].get("extra_body"), on_token=on_token)
        except Exception as e:
            print(f"  t4 record reply {i} failed: {e}")
            continue
        replies.append({"i": i, "text": result.text, "tokens": tokens})
        print(f"  recorded reply {i}: {len(tokens)} tokens")
    REPLIES_PATH.write_text(json.dumps({"replies": replies}))
    return replies


async def replay_mode(cfg: dict, log: EventLog, replies: list[dict], mode: str,
                      resume_ids: set[str]):
    tts = TTSSession(cfg["api_key"], cfg["models"]["tts"], cfg["models"]["voice"],
                     sample_rate=cfg["audio"]["tts_sample_rate"], segment=mode)
    try:
        connect_ms = await tts.connect()
    except Exception as e:
        log.log(run_id=f"t4-{mode}-000", experiment="t4", config=mode, turn=0,
                direction="meta", type_="connect.failed", meta={"error": str(e)})
        print(f"  segment={mode}: connect failed: {e}")
        return
    try:
        for rep in replies:
            run_id = f"t4-{mode}-{rep['i']:03d}"
            if run_id in resume_ids:
                continue
            sim = PlaybackSimulator(sample_rate=cfg["audio"]["tts_sample_rate"])
            state = {"t_first_send": None, "t_first_audio": None,
                     "t_done": None, "idle_since": None}

            def on_ev(ev, t_ns):
                t = ev.get("type")
                if t == "conversation.item.audio_output.delta":
                    sim.feed(t_ns, len(unb64(ev.get("delta", ""))))
                    if state["t_first_audio"] is None:
                        state["t_first_audio"] = t_ns
                    state["idle_since"] = t_ns
                elif t == "conversation.item.audio_output.done":
                    state["t_done"] = t_ns
                    state["idle_since"] = t_ns
                elif t == "conversation.item.tts.failed":
                    state["idle_since"] = t_ns
                    log.log(run_id=run_id, experiment="t4", config=mode,
                            turn=rep["i"], direction="recv", type_=t,
                            meta={"ev": ev})

            tts.on(on_ev)
            try:
                tokens = rep["tokens"] or [[rep["text"], 0.0]]
                t_start = mono_ns()
                t_loop = asyncio.get_event_loop().time()
                for text, offset_ms in tokens:
                    target = t_loop + offset_ms / 1000.0
                    now = asyncio.get_event_loop().time()
                    if target > now:
                        await asyncio.sleep(target - now)
                    await tts.append(text)
                    if state["t_first_send"] is None:
                        state["t_first_send"] = mono_ns()
                await tts.commit()
                # Read until done or 2 s of silence.
                deadline = asyncio.get_event_loop().time() + 60.0
                while asyncio.get_event_loop().time() < deadline:
                    if state["t_done"] is not None:
                        break
                    if state["idle_since"] and (mono_ns() - state["idle_since"]) > 2e9:
                        break
                    await asyncio.sleep(0.05)
                ttfa_ms = (state["t_first_audio"] - state["t_first_send"]) / 1e6 \
                    if state["t_first_audio"] and state["t_first_send"] else None
                log.log(run_id=run_id, experiment="t4", config=mode,
                        turn=rep["i"], direction="meta", type_="turn.complete",
                        meta={"ttfa_ms": ttfa_ms, "stalls": sim.stalls,
                              "n_tokens": len(tokens)})
                print(f"  {run_id}: ttfa={ttfa_ms and round(ttfa_ms)} ms, stalls={sim.stalls}")
            except Exception as e:
                log.log(run_id=run_id, experiment="t4", config=mode,
                        turn=rep["i"], direction="meta", type_="turn.failed",
                        meta={"error": str(e)})
            finally:
                tts.off(on_ev)
            await asyncio.sleep(0.5)
    finally:
        await tts.close()


async def run(cfg: dict, resume: bool = False):
    from ..config import require_key
    require_key(cfg)
    log = EventLog("t4")
    try:
        n = cfg["runs"]["t4_replies"]
        if REPLIES_PATH.exists():
            replies = json.loads(REPLIES_PATH.read_text())["replies"]
            if len(replies) < n:
                replies = []
        else:
            replies = []
        if not replies:
            llm_client = AsyncOpenAI(api_key=cfg["api_key"], base_url=BASE_URL)
            replies = await record_replies(cfg, llm_client, n)
        resume_ids = existing_run_ids("t4") if resume else set()
        for mode in MODES:
            await replay_mode(cfg, log, replies, mode, resume_ids)
    finally:
        log.close()
