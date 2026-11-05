"""clips: stitch real recorded audio at measured timings (design.md 10).

Never simulated silence from whole cloth: question audio is the actual corpus
wav, the gap is the measured t_first_audio - t_eos of the turn closest to p50,
and the reply is the actual PCM recorded during that turn.

Exports mono opus (ffmpeg required) under 200 KB each, plus a markers JSON per
clip for the page playhead.
"""
from __future__ import annotations

import io
import json
import subprocess

import numpy as np
import soundfile as sf

from .clock import AUDIO_DIR, RESULTS, read_events
from .corpus import CORPUS_DIR, item_pcm, load_manifest

CLIPS_DIR = RESULTS / "clips"
OUT_RATE = 24000  # clips at TTS rate; corpus audio resampled up


def _pcm_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def _wav_bytes(float32: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, float32, rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _to_24k(float32: np.ndarray, rate: int) -> np.ndarray:
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(OUT_RATE, rate)
    if g == OUT_RATE == rate:
        return float32
    return resample_poly(float32, OUT_RATE // g, rate // g).astype(np.float32)


def _export(name: str, audio24k: np.ndarray, markers: list[dict]) -> dict:
    """Write opus + markers JSON; returns the clip entry."""
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    src = CLIPS_DIR / f"{name}.opus"
    with open(CLIPS_DIR / f".{name}.tmp.wav", "wb") as fh:
        fh.write(_wav_bytes(audio24k, OUT_RATE))
    try:
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-i", str(CLIPS_DIR / f".{name}.tmp.wav"),
            "-ac", "1", "-c:a", "libopus", "-b:a", "32k", str(src),
        ], check=True)
    finally:
        (CLIPS_DIR / f".{name}.tmp.wav").unlink(missing_ok=True)
    entry = {"id": name, "src": f"clips/{name}.opus", "markers": markers}
    (CLIPS_DIR / f"{name}.json").write_text(json.dumps(entry, indent=2))
    kb = src.stat().st_size // 1024
    print(f"  clip {name}: {kb} KB")
    if kb > 200:
        print(f"  warning: {name}.opus exceeds 200 KB")
    return entry


def _silence(ms: float) -> np.ndarray:
    return np.zeros(int(OUT_RATE * max(ms, 0) / 1000), dtype=np.float32)


def _closest_to_p50(meta_list: list[dict], totals: list[float]) -> dict:
    med = float(np.percentile(totals, 50))
    return meta_list[int(np.argmin(np.abs(np.array(totals) - med)))]


def _turn_audio(run_id: str) -> bytes | None:
    p = AUDIO_DIR / f"{run_id}-reply-24000.pcm"
    return p.read_bytes() if p.exists() else None


def _t1_turns(cfg_name: str):
    """run_id -> (corpus set/idx info via turn.start meta, turn.complete meta)."""
    events = read_events("t1")
    starts, completes = {}, {}
    for e in events:
        if e["config"] != cfg_name:
            continue
        if e["type"] == "turn.start":
            starts[e["run_id"]] = e["meta"]
        elif e["type"] == "turn.complete" and "t_first_audio" in e.get("meta", {}):
            completes[e["run_id"]] = e["meta"]
    return starts, completes


def _vad_clip(cfg_name: str, clip_name: str, manifest: dict, rate16k: int):
    starts, completes = _t1_turns(cfg_name)
    totals, metas, ids = [], [], []
    for rid, m in completes.items():
        if m.get("t_eos") and m.get("t_first_audio") and starts.get(rid):
            totals.append((m["t_first_audio"] - m["t_eos"]) / 1e6)
            metas.append(m)
            ids.append(rid)
    if not ids:
        print(f"  {clip_name}: no complete {cfg_name} turns, skipping")
        return None
    i = int(np.argmin(np.abs(np.array(totals) - float(np.percentile(totals, 50)))))
    rid, gap_ms = ids[i], totals[i]
    corpus_id = starts[rid]["corpus"]
    set_name, _, idx = corpus_id.rpartition("-")
    q_pcm, entry = item_pcm(set_name, int(idx))
    question = _to_24k(_pcm_to_float(q_pcm), rate16k)
    reply_raw = _turn_audio(rid)
    if reply_raw is None:
        print(f"  {clip_name}: no recorded reply for {rid}, skipping")
        return None
    reply = _pcm_to_float(reply_raw)[: int(OUT_RATE * 4)]  # first 4 s of reply
    eos_pos = len(question)
    audio = np.concatenate([question, _silence(gap_ms), reply])
    return _export(clip_name, audio, [
        {"t_ms": round(eos_pos / OUT_RATE * 1000), "label": "you stop talking"},
        {"t_ms": round((eos_pos / OUT_RATE * 1000) + gap_ms), "label": "first audio"},
    ])


def _cut_off_clip(manifest: dict, rate16k: int):
    """A paused question that was cut off; the reply starts at its measured
    time, mixed over the user's second clause."""
    events = read_events("t2")
    starts, completes = {}, {}
    for e in events:
        if e["type"] == "turn.start":
            starts[e["run_id"]] = e  # full event: meta + t_ns (stream start)
        elif e["type"] == "turn.complete" and e["meta"].get("cut_off"):
            completes[e["run_id"]] = e["meta"]
    for rid, m in completes.items():
        start = starts.get(rid)
        reply_raw = _turn_audio(rid)
        if not start or reply_raw is None or not m.get("t_final"):
            continue
        set_name, _, idx = start["meta"]["corpus"].rpartition("-")
        q_pcm, entry = item_pcm(set_name, int(idx))
        question = _to_24k(_pcm_to_float(q_pcm), rate16k)
        # Clip drops the 300 ms silence pre-roll: clip time 0 = speech starts.
        t_reply = m.get("t_first_audio") or m["t_final"]
        reply_ms = (t_reply - start["t_ns"]) / 1e6 - 300.0
        start_pos = int(OUT_RATE * max(reply_ms, 0) / 1000)
        reply = _pcm_to_float(reply_raw)[: int(OUT_RATE * 4)]
        total = max(start_pos + len(reply) + int(OUT_RATE * 0.3), len(question))
        mix = np.zeros(total, dtype=np.float32)
        mix[: len(question)] += question
        mix[start_pos: start_pos + len(reply)] += reply
        peak = np.abs(mix).max() or 1.0
        if peak > 0.99:
            mix /= peak
        return _export("cut_off", mix, [
            {"t_ms": round(start_pos / OUT_RATE * 1000),
             "label": "agent answers mid-question"},
        ])
    print("  cut_off: no cut-off turn with recorded audio, skipping")
    return None


def _interrupt_clip(cfg, manifest):
    """clear_vs_cancel: reply audio; 'wait, stop' overlay at t_cmd; after that,
    clear continues for the measured audio_after_cmd_ms, cancel stops at the
    measured point. Exported as one clip per command."""
    from .clients.tts import http_tts
    import asyncio

    events = read_events("t3")
    outs = []
    for mode, clip_name in (("interrupt_clear", "clear_stops"),
                            ("interrupt_cancel", "cancel_stop")):
        evs = [e for e in events if e["config"] == mode
               and e["type"] == "turn.complete"
               and e["meta"].get("audio_after_cmd_ms") is not None]
        if not evs:
            print(f"  {clip_name}: no t3 runs, skipping")
            continue
        med = evs[len(evs) // 2]["meta"]
        # Fresh reply audio from the same interrupt text for consistent voice.
        from .experiments.t3_behavior import INTERRUPT_TEXT
        wav = asyncio.run(http_tts(cfg["api_key"], cfg["models"]["tts"],
                                   cfg["models"]["voice"], INTERRUPT_TEXT,
                                   sample_rate=24000))
        reply, _ = sf.read(io.BytesIO(wav), dtype="float32")
        if reply.ndim > 1:
            reply = reply.mean(axis=1)
        stop_wav = asyncio.run(http_tts(cfg["api_key"], cfg["models"]["tts"],
                                        cfg["models"]["voice"], "wait, stop",
                                        sample_rate=24000))
        stop, _ = sf.read(io.BytesIO(stop_wav), dtype="float32")
        if stop.ndim > 1:
            stop = stop.mean(axis=1)
        cmd_pos = int(OUT_RATE * 2.0)  # interrupt lands 2 s into the reply
        audio = list(reply[: cmd_pos])
        overlay_end = min(cmd_pos + len(stop), len(reply))
        tail = reply[cmd_pos: overlay_end] * 0.6 + stop[: overlay_end - cmd_pos]
        if mode == "interrupt_clear":
            keep_ms = med["audio_after_cmd_ms"]
            rest = reply[overlay_end: overlay_end + int(OUT_RATE * keep_ms / 1000)]
            audio += list(tail) + list(rest)
        else:
            quiet_ms = max(med.get("quiesced_ms") or 0, 0)
            rest = reply[overlay_end: overlay_end + int(OUT_RATE * quiet_ms / 1000)]
            audio += list(tail) + list(rest) + list(_silence(400))
        mix = np.array(audio, dtype=np.float32)
        peak = np.abs(mix).max() or 1.0
        if peak > 0.99:
            mix /= peak
        outs.append(_export(clip_name, mix, [
            {"t_ms": round(cmd_pos / OUT_RATE * 1000), "label": "wait, stop"},
        ]))
    return outs


def run(cfg: dict):
    manifest = load_manifest()
    rate16k = cfg["audio"]["stt_sample_rate"]
    made = []
    c = _vad_clip("default", "vad_default", manifest, rate16k)
    made += [c] if c else []
    c = _vad_clip("tuned", "vad_tuned", manifest, rate16k)
    made += [c] if c else []
    c = _cut_off_clip(manifest, rate16k)
    made += [c] if c else []
    made += _interrupt_clip(cfg, manifest)
    print(f"clips: {len(made)} written to {CLIPS_DIR}")
