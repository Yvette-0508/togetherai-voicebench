#!/usr/bin/env python3
"""design.md §14 "verify first" pass.

Confirms every §6 API-surface assumption against the live Together API before
running the P0/T0-T4 experiments. Prints raw messages per event name, then a
PASS/FAIL summary. Results land in results/verify.json (numbers from this file
feed the report page, e.g. the context.cancel check).
"""
from __future__ import annotations

import asyncio
import io
import json
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from voicebench.clients.stt import STTSession  # noqa: E402
from voicebench.clients.tts import TTSSession, TTS_HTTP_URL  # noqa: E402
from voicebench.clock import (FramePacer, chunk_pcm, mono_ns, silence_frame)  # noqa: E402
from voicebench.config import load_config, require_key  # noqa: E402

FRAME_MS = 20
SR = 16000
UTTERANCE_TEXT = "What's the weather like in San Francisco right now?"
RESULTS = Path(__file__).resolve().parent.parent / "results"

RAW: dict[str, dict] = {}          # stage -> {event_type: {"example": ev, "count": n}}
CHECKS: dict[str, dict] = {}       # check name -> result


def record(stage: str):
    def cb(ev, _t_ns):
        t = ev.get("type", "?")
        slot = RAW.setdefault(stage, {}).setdefault(t, {"example": ev, "count": 0})
        slot["count"] += 1
    return cb


def trim(ev, limit: int = 120):
    def t(v):
        if isinstance(v, str) and len(v) > limit:
            return v[:limit] + f"...<+{len(v) - limit} chars>"
        if isinstance(v, dict):
            return {k: t(x) for k, x in v.items()}
        if isinstance(v, list):
            return [t(x) for x in v]
        return v
    return t(ev)


async def pace_frames(sess, frames, tail_count=0):
    """Send frames on a ~real-time schedule; None = silence frame."""
    for f in frames:
        await sess.append(f if f is not None else silence_frame(FRAME_MS, SR))
        await asyncio.sleep(FRAME_MS / 1000)


def summary(name: str, ok: bool, evidence: str, **extra):
    CHECKS[name] = {"pass": ok, "evidence": evidence, **extra}
    print(f"\n[{'PASS' if ok else 'FAIL'}] {name}\n      {evidence}")


async def gen_utterance(cfg) -> bytes:
    """HTTP TTS at 16 kHz -> PCM s16le. Also serves as the sample_rate=16000 check."""
    cfg16 = dict(cfg)
    text = UTTERANCE_TEXT
    headers = {"Authorization": f"Bearer {cfg['api_key']}"}
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(TTS_HTTP_URL, headers=headers, json={
            "model": cfg["models"]["tts"], "voice": cfg["models"]["voice"],
            "input": text, "response_format": "wav", "sample_rate": SR,
        })
        r.raise_for_status()
        RAW.setdefault("http_tts", {})["response_headers"] = {
            "example": {k: v for k, v in r.headers.items()
                        if k.lower().startswith(("x-", "retry", "cf-"))},
            "count": 1}
        CHECKS["tts_sample_rate_16000"] = {}
        with wave.open(io.BytesIO(r.content)) as w:
            rate = w.getframerate()
            pcm = w.readframes(w.getnframes())
        summary("tts_sample_rate_16000", rate == SR,
                f"HTTP TTS with sample_rate=16000 returned a {rate} Hz wav "
                f"({'no resampling needed for corpus' if rate == SR else 'corpus must keep resampling'})")
    # HTTP TTS ignores sample_rate=16000 and returns 24 kHz; resample to 16 k
    # for the STT checks (and proves corpus.py must keep its resample step).
    if rate != SR:
        import numpy as np
        from scipy.signal import resample_poly
        x = np.frombuffer(pcm, dtype=np.int16)
        pcm = resample_poly(x, SR, rate).round().astype(np.int16).tobytes()
    return pcm


async def send_silence(sess, f, i, t):
    """FramePacer on_frame: None -> real silence frame."""
    await sess.append(f if f is not None else silence_frame(FRAME_MS, SR))


async def check_vad_on(cfg, pcm):
    sess = STTSession(cfg["api_key"], cfg["models"]["stt"],
                      vad=True, silence_ms=cfg["vad"]["default_silence_ms"])
    sess.on(record("stt_vad_on"))
    await sess.connect()
    got = {}
    sess.on(lambda ev, t: got.setdefault("completed", (ev, t))
            if ev["type"] == "conversation.item.input_audio_transcription.completed" else None)

    frames = chunk_pcm(pcm, SR, FRAME_MS)
    pacer = FramePacer(FRAME_MS)
    t_speech_end, _ = await pacer.run(frames, lambda f, i, t: send_silence(sess, f, i, t),
                                      should_continue=lambda: "completed" not in got)
    done_evt = got.get("completed")
    if done_evt:
        delta = (done_evt[1] - t_speech_end) / 1e6
        summary("vad_on_completed_without_commit", True,
                f"completed arrived {delta:.0f} ms after speech end, no commit sent",
                ms_after_speech_end=round(delta))
    else:
        summary("vad_on_completed_without_commit", False,
                "no completed event within silence tail")
    await sess.close()


async def check_vad_off(cfg, pcm):
    sess = STTSession(cfg["api_key"], cfg["models"]["stt"], vad=False)
    sess.on(record("stt_vad_off"))
    await sess.connect()
    got = []
    sess.on(lambda ev, t: got.append(t)
            if ev["type"] == "conversation.item.input_audio_transcription.completed" else None)

    frames = chunk_pcm(pcm, SR, FRAME_MS)
    tail_frames = 125  # 2.5 s of silence
    pacer = FramePacer(FRAME_MS)
    tail = {"n": 0}
    async def send(f, i, t):
        await sess.append(f if f is not None else None or silence_frame(FRAME_MS, SR))
    # stream speech + manual 2.5 s silence tail
    await pace_frames(sess, frames + [None] * tail_frames)

    completed_before_commit = bool(got)
    t_commit = mono_ns()
    await sess.commit()
    try:
        ev, t = await sess.wait_for(
            "conversation.item.input_audio_transcription.completed",
            since_ns=t_commit, timeout=8.0)
        summary("vad_off_completed_after_commit", not completed_before_commit,
                f"no completed during 2.5 s tail; completed arrived "
                f"{(t - t_commit) / 1e6:.0f} ms after commit",
                completed_before_commit=completed_before_commit)
    except asyncio.TimeoutError:
        summary("vad_off_completed_after_commit", False,
                f"no completed within 8 s of commit "
                f"(before_commit={completed_before_commit})")
    await sess.close()


async def check_vad_mid_session_update(cfg, pcm):
    """Raise min_silence 500 -> 1500 mid-session; 900 ms of silence must NOT
    trigger completion, then more silence must."""
    sess = STTSession(cfg["api_key"], cfg["models"]["stt"], vad=True, silence_ms=500)
    sess.on(record("stt_vad_update"))
    await sess.connect()
    await sess.update_vad(silence_ms=1500)
    got = []
    sess.on(lambda ev, t: got.append(t)
            if ev["type"] == "conversation.item.input_audio_transcription.completed" else None)

    frames = chunk_pcm(pcm, SR, FRAME_MS)
    await pace_frames(sess, frames)
    t_speech_end = mono_ns()
    await pace_frames(sess, [None] * 45)      # +900 ms silence
    if got:
        summary("vad_mid_session_update", False,
                f"completed fired ~{(got[0] - t_speech_end) / 1e6:.0f} ms after speech end "
                f"with only 900 ms of silence streamed: 1500 ms update had no effect")
        await sess.close()
        return
    t_wait = mono_ns()
    await asyncio.sleep(0.6)                  # total post-speech wait > 1500 ms of stream time
    if got:
        summary("vad_mid_session_update", True,
                f"no completed during 900 ms silence tail; completed arrived "
                f"{(got[0] - t_speech_end) / 1e6:.0f} ms after speech end once "
                f"streamed silence exceeded 1500 ms (update took effect)",
                ms_after_speech_end=round((got[0] - t_speech_end) / 1e6))
    else:
        await pace_frames(sess, [None] * 40)  # +800 ms more, just in case
        await asyncio.sleep(1.0)
        summary("vad_mid_session_update", bool(got),
                (f"completed {(got[0] - t_speech_end) / 1e6:.0f} ms after speech end"
                 if got else "still no completed after ~1.7 s of streamed silence"),
                ms_after_speech_end=round((got[0] - t_speech_end) / 1e6) if got else None)
    await sess.close()


async def check_context_cancel(cfg):
    """append without context_id, then context.cancel on the 'default' context."""
    sess = TTSSession(cfg["api_key"], cfg["models"]["tts"], cfg["models"]["voice"],
                      sample_rate=24000)
    sess.on(record("tts_cancel"))
    await sess.connect()
    deltas = []

    def on_delta(ev, t):
        if ev["type"] == "conversation.item.audio_output.delta":
            deltas.append((t, len(ev.get("delta", ""))))
    sess.on(on_delta)

    long_text = ("One two three four five six seven eight nine ten. " * 8)
    t_append = mono_ns()
    await sess.append(long_text)              # no context_id
    await sess.commit()

    # wait for first audio, let ~300 ms of audio stream in, then cancel
    try:
        await sess.wait_for("conversation.item.audio_output.delta",
                            since_ns=t_append, timeout=15.0)
    except asyncio.TimeoutError:
        summary("context_cancel_default", False, "no audio deltas to cancel")
        await sess.close()
        return
    await asyncio.sleep(0.3)
    t_cancel = mono_ns()
    n_before = len(deltas)
    await sess.cancel()                       # context_id defaults to "default"
    ok_cancelled = True
    try:
        ev, t = await sess.wait_for("context.cancelled", since_ns=t_cancel, timeout=5.0)
    except asyncio.TimeoutError:
        ok_cancelled = False
        t = mono_ns()
    await asyncio.sleep(1.0)
    n_after = len(deltas)
    t_last_delta = deltas[-1][0] if deltas else t_cancel

    summary("context_cancel_default", ok_cancelled,
            f"context.cancelled {'received' if ok_cancelled else 'NOT received'} "
            f"{(t - t_cancel) / 1e6:.0f} ms after cancel; deltas before cancel: {n_before}, "
            f"after: {n_after} (last {(t_last_delta - t_cancel) / 1e6:.0f} ms post-cancel)",
            deltas_before_cancel=n_before, deltas_after_cancel=n_after)
    await sess.close()


async def check_tts_delivery_speed(cfg):
    sess = TTSSession(cfg["api_key"], cfg["models"]["tts"], cfg["models"]["voice"],
                      sample_rate=24000)
    sess.on(record("tts_speed"))
    await sess.connect()
    first = {"t": None}
    audio_ms = {"v": 0.0}

    def on_delta(ev, t):
        bytes_ = len(ev.get("delta", "")) * 3 // 4
        ev_ms = bytes_ / 2 / 24.0  # 24 kHz 16-bit mono
        audio_ms["v"] += ev_ms
        if ev["type"] == "conversation.item.audio_output.delta" and first["t"] is None:
            first["t"] = t
    sess.on(on_delta)

    t0 = mono_ns()
    await sess.append("The quick brown fox jumps over the lazy dog, and then "
                      "keeps running until it is well and truly out of breath.")
    await sess.commit()
    try:
        await sess.wait_for("conversation.item.audio_output.done", since_ns=t0, timeout=20.0)
    except asyncio.TimeoutError:
        pass
    t_done = mono_ns()
    wall_ms = (t_done - t0) / 1e6
    ratio = audio_ms["v"] / wall_ms if wall_ms else 0
    summary("tts_delivery_speed", True,
            f"{audio_ms['v']:.0f} ms of audio delivered in {wall_ms:.0f} ms "
            f"({ratio:.1f}x real time); first delta "
            f"{((first['t'] or t_done) - t0) / 1e6:.0f} ms after append",
            audio_ms=round(audio_ms["v"]), wall_ms=round(wall_ms),
            speed_ratio=round(ratio, 2))
    await sess.close()


async def check_http_streaming(cfg):
    chunk_info = {"n": 0, "bytes": 0, "t_first": None}
    t0 = mono_ns()
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST", TTS_HTTP_URL,
            headers={"Authorization": f"Bearer {cfg['api_key']}"},
            json={"model": cfg["models"]["tts"], "voice": cfg["models"]["voice"],
                  "input": "Streaming test sentence.", "response_format": "raw",
                  "sample_rate": 24000, "stream": True},
        ) as r:
            r.raise_for_status()
            RAW.setdefault("http_stream", {})["response_headers"] = {
                "example": {k: v for k, v in r.headers.items()
                            if k.lower().startswith(("content", "transfer", "x-"))},
                "count": 1}
            async for chunk in r.aiter_bytes():
                if chunk_info["t_first"] is None:
                    chunk_info["t_first"] = mono_ns()
                chunk_info["n"] += 1
                chunk_info["bytes"] += len(chunk)
    t_first = (chunk_info["t_first"] or t0) - t0
    summary("http_streaming_variant", chunk_info["n"] > 0,
            f"response_format=raw + stream=true: {chunk_info['n']} chunks, "
            f"{chunk_info['bytes']} bytes total, first chunk after {t_first / 1e6:.0f} ms; "
            f"content-type "
            f"{RAW['http_stream']['response_headers']['example'].get('content-type', '?')}",
            chunks=chunk_info["n"], bytes=chunk_info["bytes"],
            first_chunk_ms=round(t_first / 1e6))


async def check_llm_model(cfg):
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            "https://api.together.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {cfg['api_key']}"},
            json={"model": cfg["models"]["llm"],
                  "messages": [{"role": "user", "content": "Reply with: ok"}],
                  "max_tokens": 8, "stream": False,
                  **(cfg.get("llm", {}).get("extra_body") or {})})
        ok = r.status_code == 200
        body = r.json() if ok else r.text[:200]
        summary("llm_model_available", ok,
                f"HTTP {r.status_code} for {cfg['models']['llm']}"
                + (f" -> {body['choices'][0]['message']['content']!r}" if ok else f" -> {body}"))


async def main():
    cfg = load_config()
    require_key(cfg)
    errors_429 = 0

    print("== §14 verify pass ==\nfacts: generating test utterance (also checks sample_rate=16000)")
    try:
        pcm = await gen_utterance(cfg)
    except Exception as e:  # capture but continue
        print(f"  !! utterance generation failed: {e}")
        pcm = b"\x00" * int(1.5 * SR * 2)  # silence so STT checks still exercise events

    checks = [
        ("1  VAD on: completed without commit", lambda: check_vad_on(cfg, pcm)),
        ("2  VAD off: completed only after commit", lambda: check_vad_off(cfg, pcm)),
        ("3  transcription_session.updated mid-session", lambda: check_vad_mid_session_update(cfg, pcm)),
        ("5  context.cancel on default context", lambda: check_context_cancel(cfg)),
        ("6  TTS delivery speed", lambda: check_tts_delivery_speed(cfg)),
        ("7  HTTP streaming TTS variant", lambda: check_http_streaming(cfg)),
        ("+  LLM model availability", lambda: check_llm_model(cfg)),
    ]
    for label, fn in checks:
        print(f"\n== check {label}")
        try:
            await fn()
        except Exception as e:
            if "429" in str(e):
                errors_429 += 1
            summary(label.split(" ", 1)[1][:40], False, f"exception: {type(e).__name__}: {e}")

    summary("rate_limits_429s", True,
            f"{errors_429} HTTP 429s observed during the verify pass",
            count_429=errors_429)

    # ---- raw event examples (design §14 line 1) ----
    print("\n== raw events seen (one example per §6 occurrence) ==")
    examples = {}
    for stage, events in RAW.items():
        for t, slot in events.items():
            examples.setdefault(t, slot["example"])
            print(f"  {stage}: {t}  x{slot['count']}")
    # ---- persist ----
    RESULTS.mkdir(exist_ok=True)
    out = {"checks": CHECKS,
           "raw_events": {t: trim(ev) for t, ev in examples.items()},
           "config": {"models": cfg["models"], "vad": cfg["vad"],
                      "client_location": cfg.get("client_location")}}
    (RESULTS / "verify.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    n_pass = sum(1 for c in CHECKS.values() if c.get("pass"))
    print(f"\n== summary: {n_pass}/{len(CHECKS)} checks passed; "
          f"raw events logged to results/verify.json ==")
    sys.exit(0 if n_pass == len(CHECKS) else 1)


if __name__ == "__main__":
    asyncio.run(main())
