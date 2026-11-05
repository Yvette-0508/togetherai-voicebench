"""T3. Agent behavior checks (design.md section 9, P0).

Implemented here:
- long_question: stream an 8-9 s item with default VAD; pass iff exactly one
  `completed` event (i.e. it stays one turn).
- openai_voice: OpenAI SDK audio.speech.create against the Together base URL
  with voice="alloy"; pass iff HTTP 200 with audio.
- interruption: input_text_buffer.clear / context.cancel per the interruption
  procedure; records t_cmd, t_cancelled, t_last_delta, audio_after_cmd_ms and
  buffered_at_cmd_ms (playback simulator).

Computed in summarize from T1 logs instead: tool_valid, wer.
"""
from __future__ import annotations

import asyncio

from openai import AsyncOpenAI

from ..clients.stt import STTSession
from ..clients.tts import TTSSession
from ..clock import (EventLog, FramePacer, chunk_pcm, existing_run_ids,
                     mono_ns, silence_frame, unb64)
from ..corpus import item_pcm
from ..playback import PlaybackSimulator
from .t1_turn_latency import BASE_URL

INTERRUPT_TEXT = (
    "Sure, here is a longer answer that takes a while to speak. "
    "The first thing to know is that timing in voice interfaces matters a lot. "
    "A reply like this one keeps going for quite a few seconds on purpose. "
    "By the time it finishes, a real user would have long since interrupted."
)


async def _long_question(cfg, log, resume_ids):
    n = cfg["runs"]["t3_runs"]
    stt = STTSession(cfg["api_key"], cfg["models"]["stt"], vad=True,
                     silence_ms=cfg["vad"]["default_silence_ms"],
                     sample_rate=cfg["audio"]["stt_sample_rate"])
    await stt.connect()
    try:
        for i in range(n):
            run_id = f"t3-long-{i:03d}"
            if run_id in resume_ids:
                continue
            pcm, entry = item_pcm("long", i % 12)
            frames = chunk_pcm(pcm, cfg["audio"]["stt_sample_rate"],
                               cfg["audio"]["frame_ms"])
            n_completed = {"n": 0}

            def on_ev(ev, t_ns):
                if ev.get("type") == "conversation.item.input_audio_transcription.completed":
                    n_completed["n"] += 1

            stt.on(on_ev)
            log.log(run_id=run_id, experiment="t3", config="long_question",
                    turn=i, direction="meta", type_="turn.start",
                    meta={"corpus": entry["id"]})
            try:
                pacer = FramePacer(cfg["audio"]["frame_ms"])
                async def send(frame, k, t_ns):
                    await stt.append(frame if frame is not None else
                                     silence_frame(cfg["audio"]["frame_ms"],
                                                   cfg["audio"]["stt_sample_rate"]))
                # Long items can exceed the 3 s tail; give VAD time to finish.
                tail_silent = int(3.0 / (cfg["audio"]["frame_ms"] / 1000))
                for fr in frames:
                    await send(fr, 0, mono_ns())
                    await asyncio.sleep(cfg["audio"]["frame_ms"] / 1000)
                for _ in range(tail_silent):
                    await send(None, 0, mono_ns())
                    await asyncio.sleep(cfg["audio"]["frame_ms"] / 1000)
                await asyncio.sleep(2.0)  # let late completions arrive
                log.log(run_id=run_id, experiment="t3", config="long_question",
                        turn=i, direction="meta", type_="turn.complete",
                        meta={"completed_events": n_completed["n"],
                              "pass": n_completed["n"] == 1})
            except Exception as e:
                log.log(run_id=run_id, experiment="t3", config="long_question",
                        turn=i, direction="meta", type_="turn.failed",
                        meta={"error": str(e)})
            finally:
                stt.off(on_ev)
            await asyncio.sleep(1.0)
    finally:
        await stt.close()


async def _openai_voice(cfg, log, resume_ids):
    client = AsyncOpenAI(api_key=cfg["api_key"], base_url=BASE_URL)
    for i in range(3):
        run_id = f"t3-openai-voice-{i:03d}"
        if run_id in resume_ids:
            continue
        status, nbytes, err = 0, 0, None
        try:
            resp = await client.audio.speech.create(
                model=cfg["models"]["tts"], voice="alloy",
                input="Hello, this is a compatibility check.")
            nbytes = len(resp.content)
            status = 200 if nbytes > 0 else 0
        except Exception as e:
            status = getattr(e, "status_code", 0) or 0
            err = str(e)[:300]
        log.log(run_id=run_id, experiment="t3", config="openai_voice", turn=i,
                direction="meta", type_="check.complete",
                meta={"status": status, "audio_bytes": nbytes,
                      "error": err, "pass": status == 200 and nbytes > 0})


async def _interruption_one(cfg, log, run_id, i, *, mode: str):
    """mode: 'clear' or 'cancel'. Interruption procedure, design.md section 9."""
    sr = cfg["audio"]["tts_sample_rate"]
    tts = TTSSession(cfg["api_key"], cfg["models"]["tts"], cfg["models"]["voice"],
                     sample_rate=sr)
    await tts.connect()
    sim = PlaybackSimulator(sample_rate=sr)
    state = {"t_first_delta": None, "t_cmd": None, "t_last_delta": None,
             "t_cancelled": None, "audio_after_ms": 0.0, "stop": False}

    def on_ev(ev, t_ns):
        t = ev.get("type")
        if t == "conversation.item.audio_output.delta":
            chunk = unb64(ev.get("delta", ""))
            sim.feed(t_ns, len(chunk))
            if state["t_first_delta"] is None:
                state["t_first_delta"] = t_ns
            state["t_last_delta"] = t_ns
            if state["t_cmd"] is not None and t_ns >= state["t_cmd"]:
                state["audio_after_ms"] += sim.dur_ms(len(chunk))
        elif t == "context.cancelled":
            state["t_cancelled"] = t_ns

    tts.on(on_ev)
    try:
        ctx = "reply" if mode == "cancel" else None
        if ctx:
            await tts.append(INTERRUPT_TEXT, context_id=ctx)
        else:
            await tts.append(INTERRUPT_TEXT)
        await tts.commit()
        # Wait for the first delta (+ up to 10 s), then send the command 500 ms later.
        try:
            while state["t_first_delta"] is None:
                await asyncio.sleep(0.01)
        finally:
            pass
        await asyncio.sleep(0.5)
        buffered_ms = sim.buffered_ahead_ms(mono_ns())
        state["t_cmd"] = mono_ns()
        if mode == "clear":
            await tts.clear()
        else:
            await tts.cancel(ctx or "default")
        # Keep reading until there are 2 s with no deltas.
        deadline = asyncio.get_event_loop().time() + 30.0
        while asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.05)
            last = state["t_last_delta"]
            now = mono_ns()
            if last is not None and (now - last) > 2e9:
                break
        after_cmd_ms = None
        if state["t_last_delta"] and state["t_cmd"]:
            after_cmd_ms = (state["t_last_delta"] - state["t_cmd"]) / 1e6
        quiet_ms = (state["t_cancelled"] or state["t_last_delta"] or 0) - (state["t_cmd"] or 0)
        log.log(run_id=run_id, experiment="t3", config=f"interrupt_{mode}",
                turn=i, direction="meta", type_="turn.complete",
                meta={
                    "t_cmd": state["t_cmd"], "t_cancelled": state["t_cancelled"],
                    "t_last_delta": state["t_last_delta"],
                    "audio_after_cmd_ms": round(state["audio_after_ms"], 1),
                    "buffered_at_cmd_ms": round(buffered_ms, 1),
                    "last_delta_after_cmd_ms": (after_cmd_ms if after_cmd_ms is None
                                                else round(after_cmd_ms, 1)),
                    "quiesced_ms": round(quiet_ms / 1e6, 1),
                })
    finally:
        await tts.close()


async def _interruption(cfg, log, resume_ids):
    n = cfg["runs"]["t3_runs"]
    for mode in ("clear", "cancel"):
        for i in range(n):
            run_id = f"t3-{mode}-{i:03d}"
            if run_id in resume_ids:
                continue
            try:
                await asyncio.wait_for(
                    _interruption_one(cfg, log, run_id, i, mode=mode), timeout=60)
            except Exception as e:
                log.log(run_id=run_id, experiment="t3",
                        config=f"interrupt_{mode}", turn=i, direction="meta",
                        type_="turn.failed", meta={"error": str(e)})
            await asyncio.sleep(0.5)


async def run(cfg: dict, resume: bool = False):
    from ..config import require_key
    require_key(cfg)
    log = EventLog("t3")
    try:
        resume_ids = existing_run_ids("t3") if resume else set()
        await _long_question(cfg, log, resume_ids)
        await _openai_voice(cfg, log, resume_ids)
        await _interruption(cfg, log, resume_ids)
    finally:
        log.close()
