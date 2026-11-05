"""T1. Turn latency (design.md section 9, P0).

One voice turn: stream corpus audio in real time through STT -> streamed chat
completion -> TTS, logging every event with a monotonic timestamp.

Configs (2 warm-up turns each, discarded, then N measured):
  default: VAD on, silence 500 ms, plain answer, corpus "short"
  tuned:   VAD on, silence 250 ms, plain answer, corpus "short"
  tool:    VAD on, silence 250 ms, one tool call then answer, corpus "tool"
  control: VAD off; commit sent at t_eos, corpus "short"

Timestamps per turn: t_eos, t_final, t_first_sentence, t_first_sentence_sent,
t_first_audio. run_turn() is shared with T2.
"""
from __future__ import annotations

import asyncio
import traceback
from pathlib import Path

from openai import AsyncOpenAI

from ..clients.llm import WEATHER_TOOL, stream_chat
from ..clients.stt import STTSession
from ..clients.tts import TTSSession
from ..clock import (AUDIO_DIR, EventLog, FramePacer, chunk_pcm, existing_run_ids,
                     mono_ns, save_pcm, silence_frame)
from ..corpus import item_pcm, load_manifest

BASE_URL = "https://api.together.ai/v1"


def config_table(cfg: dict) -> dict[str, dict]:
    v = cfg["vad"]
    return {
        "default": {"vad": True, "silence_ms": v["default_silence_ms"], "set": "short", "tools": False},
        "tuned": {"vad": True, "silence_ms": v["tuned_silence_ms"], "set": "short", "tools": False},
        "tool": {"vad": True, "silence_ms": v["tuned_silence_ms"], "set": "tool", "tools": True},
        "control": {"vad": False, "silence_ms": 0, "set": "short", "tools": False, "commit": True},
    }


async def run_turn(cfg: dict, stt: STTSession, tts: TTSSession,
                   llm_client: AsyncOpenAI, set_name: str, idx: int,
                   log: EventLog, run_id: str, config: str, turn: int,
                   *, tools: bool = False, commit_at_eos: bool = False,
                   experiment: str = "t1", save_audio: bool = True) -> dict:
    """Run one full voice turn. Returns the measured timestamps (ns) in a dict."""
    sr = cfg["audio"]["stt_sample_rate"]
    frame_ms = cfg["audio"]["frame_ms"]
    pcm, entry = item_pcm(set_name, idx)
    frames = chunk_pcm(pcm, sr, frame_ms)
    n_speech = len(frames)
    tts_rate = cfg["audio"]["tts_sample_rate"]

    state = {
        "t_eos": None, "t_final": None, "final_text": None,
        "t_first_sentence": None, "t_first_sentence_sent": None,
        "t_first_audio": None, "tts_done": None, "cut_off": False,
    }
    turn_start = mono_ns()
    final_evt = asyncio.Event()
    tts_done_evt = asyncio.Event()
    reply_pcm = bytearray()

    def on_stt(ev, t_ns):
        t = ev.get("type")
        if t in ("conversation.item.input_audio_transcription.completed",
                 "conversation.item.input_audio_transcription.delta"):
            text = ev.get("transcript") or ev.get("delta") or ev.get("text") or ""
            meta = {"chars": len(str(text))}
            if t.endswith("completed"):
                meta["text"] = str(text)
                if state["t_final"] is None:
                    state["t_final"] = t_ns
                    state["final_text"] = str(text)
                    if state["t_eos"] is None or t_ns < state["t_eos"]:
                        state["cut_off"] = True
                    final_evt.set()
            log.log(run_id=run_id, experiment=experiment, config=config,
                    turn=turn, direction="recv", type_=t, meta=meta)

    def on_tts(ev, t_ns):
        t = ev.get("type")
        if t == "conversation.item.audio_output.delta":
            from ..clock import unb64
            chunk = unb64(ev.get("delta", ""))
            reply_pcm.extend(chunk)
            if state["t_first_audio"] is None:
                state["t_first_audio"] = t_ns
                log.log(run_id=run_id, experiment=experiment, config=config,
                        turn=turn, direction="recv", type_="tts.first_audio",
                        meta={"bytes": len(chunk)})
        elif t == "conversation.item.audio_output.done":
            state["tts_done"] = t_ns
            tts_done_evt.set()
        elif t == "conversation.item.tts.failed":
            log.log(run_id=run_id, experiment=experiment, config=config,
                    turn=turn, direction="recv", type_=t, meta={"ev": ev})

    stt.on(on_stt)
    tts.on(on_tts)
    log.log(run_id=run_id, experiment=experiment, config=config, turn=turn,
            direction="meta", type_="turn.start",
            meta={"corpus": entry["id"], "text": entry["text"],
                  "n_speech_frames": n_speech})
    try:
        async def on_frame(frame: bytes | None, k: int, t_ns: int):
            if frame is None:
                await stt.append(silence_frame(frame_ms, sr))
                return
            await stt.append(frame)
            if k == n_speech - 1:  # last speech frame just left
                state["t_eos"] = t_ns
                log.log(run_id=run_id, experiment=experiment, config=config,
                        turn=turn, direction="send", type_="stream.eos", meta={})
                if commit_at_eos:
                    await stt.commit()
                    log.log(run_id=run_id, experiment=experiment, config=config,
                            turn=turn, direction="send",
                            type_="input_audio_buffer.commit", meta={})

        pacer = FramePacer(frame_ms)
        await pacer.run(frames, on_frame,
                        should_continue=lambda: state["t_final"] is None,
                        max_tail_s=3.0)

        # Wait for the final transcript (tail may have timed out first).
        try:
            await asyncio.wait_for(final_evt.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            raise RuntimeError("no final transcript")

        if state["t_eos"] is None:
            state["t_eos"] = state["t_final"]  # degenerate: never sent eos

        sent_first = {"done": False}

        async def send_sentence(sent: str):
            t_sent0 = mono_ns()
            await tts.append(sent)
            t = mono_ns()
            if not sent_first["done"]:
                sent_first["done"] = True
                state["t_first_sentence_sent"] = t
            log.log(run_id=run_id, experiment=experiment, config=config,
                    turn=turn, direction="send", type_="input_text_buffer.append",
                    meta={"chars": len(sent), "text": sent,
                          "dt_ms": (t - t_sent0) / 1e6})

        llm_ev_counts: dict[str, int] = {}

        def on_llm_event(name, meta):
            llm_ev_counts[name] = llm_ev_counts.get(name, 0) + 1
            log.log(run_id=run_id, experiment=experiment, config=config,
                    turn=turn, direction="recv", type_=f"llm.{name}", meta=meta)

        result = await stream_chat(
            llm_client, cfg["models"]["llm"],
            cfg["llm"]["system"], state["final_text"],
            max_tokens=cfg["llm"]["max_tokens"], temperature=cfg["llm"]["temperature"],
            tools=[WEATHER_TOOL] if tools else None,
            on_sentence=send_sentence, on_event=on_llm_event)

        if result.sentences:
            state["t_first_sentence"] = result.t_first_sentence
        if result.tool_calls:
            log.log(run_id=run_id, experiment=experiment, config=config,
                    turn=turn, direction="meta", type_="llm.tool_call",
                    meta={"calls": result.tool_calls, "valid": result.tool_valid})
        await tts.commit()
        log.log(run_id=run_id, experiment=experiment, config=config, turn=turn,
                direction="send", type_="input_text_buffer.commit", meta={})

        try:
            await asyncio.wait_for(tts_done_evt.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            pass  # first_audio is what we measure; done just ends cleanly

        if save_audio and reply_pcm:
            save_pcm(AUDIO_DIR / f"{run_id}-reply-{tts_rate}.pcm", bytes(reply_pcm))

        log.log(run_id=run_id, experiment=experiment, config=config, turn=turn,
                direction="meta", type_="turn.complete",
                meta={k: state[k] for k in
                      ("t_eos", "t_final", "t_first_sentence",
                       "t_first_sentence_sent", "t_first_audio", "cut_off")})
        return state
    finally:
        stt.off(on_stt)
        tts.off(on_tts)


async def run_config(cfg: dict, llm_client: AsyncOpenAI, name: str, spec: dict,
                     log: EventLog, n_turns: int, resume_ids: set[str],
                     experiment: str = "t1", run_id_prefix: str | None = None):
    """Warm sessions + warm-up turns + measured turns for one config."""
    log_cfg = None
    prefix = run_id_prefix or f"{experiment}-{name}"
    # Drop already-measured run ids (--resume).
    skip = {int(r.rsplit("-", 1)[1]) for r in resume_ids
            if r.startswith(prefix + "-") and r.rsplit("-", 1)[1].isdigit()}
    todo = [i for i in range(n_turns) if i not in skip]
    if not todo:
        print(f"  {prefix}: all {n_turns} turns already logged, skipping")
        return
    stt = STTSession(cfg["api_key"], cfg["models"]["stt"], vad=spec["vad"],
                     silence_ms=spec["silence_ms"],
                     sample_rate=cfg["audio"]["stt_sample_rate"])
    tts = TTSSession(cfg["api_key"], cfg["models"]["tts"], cfg["models"]["voice"],
                     sample_rate=cfg["audio"]["tts_sample_rate"])
    await stt.connect()
    await tts.connect()
    try:
        manifest = load_manifest()
        n_items = len(manifest["sets"][spec["set"]])
        for i in range(cfg["runs"]["warmup"]):  # warm-up, discarded (not logged)
            try:
                warmup_spec = dict(spec)  # warm the same path as measured turns
                wlog = EventLog("_warmup")
                await run_turn(cfg, stt, tts, llm_client, spec["set"],
                               (i % n_items), wlog, f"{prefix}-warmup-{i}",
                               name, i, tools=spec["tools"],
                               commit_at_eos=bool(spec.get("commit")),
                               experiment=experiment, save_audio=False)
                wlog.close()
            except Exception as e:
                print(f"  warmup {prefix} turn {i} failed: {e}")
        for seq, i in enumerate(todo):
            run_id = f"{prefix}-{i:03d}"
            try:
                await run_turn(cfg, stt, tts, llm_client, spec["set"],
                               (i % n_items), log, run_id, name, i,
                               tools=spec["tools"],
                               commit_at_eos=bool(spec.get("commit")),
                               experiment=experiment)
                print(f"  {run_id} ok")
            except Exception as e:
                log.log(run_id=run_id, experiment=experiment, config=name,
                        turn=i, direction="meta", type_="turn.failed",
                        meta={"error": str(e),
                              "trace": traceback.format_exc()[-800:]})
                print(f"  {run_id} FAILED: {e}")
            await asyncio.sleep(1.0)  # settle the VAD between turns
    finally:
        await stt.close()
        await tts.close()


async def run(cfg: dict, resume: bool = False):
    from ..config import require_key
    require_key(cfg)
    log = EventLog("t1")
    try:
        llm_client = AsyncOpenAI(api_key=cfg["api_key"], base_url=BASE_URL)
        resume_ids = existing_run_ids("t1") if resume else set()
        n = cfg["runs"]["t1_turns"]
        for name, spec in config_table(cfg).items():
            await run_config(cfg, llm_client, name, spec, log, n, resume_ids)
    finally:
        log.close()
