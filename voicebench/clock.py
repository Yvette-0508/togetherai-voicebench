"""Monotonic clock, real-time frame pacer, JSONL event log.

All timestamps in the harness come from time.perf_counter_ns() (mono_ns()).
Every sent/received protocol event is appended to results/raw/{experiment}.jsonl
(voicebench.section 7.1 of design.md).
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import threading
import time
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
RAW_DIR = RESULTS / "raw"
AUDIO_DIR = RESULTS / "audio"


def mono_ns() -> int:
    return time.perf_counter_ns()


def ns_to_ms(ns: int) -> float:
    return ns / 1e6


class EventLog:
    """Append-only JSONL event log. One event per line, design.md 7.1.

    {
      "run_id": "t1-default-007", "experiment": "t1", "config": "default",
      "turn": 7, "t_ns": 123456789, "dir": "recv",
      "type": "conversation.item.input_audio_transcription.completed",
      "meta": {"chars": 41}
    }

    Audio payloads are never logged; audio is saved separately when needed.
    """

    def __init__(self, experiment: str):
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        self.path = RAW_DIR / f"{experiment}.jsonl"
        self._fh = open(self.path, "a", buffering=1, encoding="utf-8")
        self._lock = threading.Lock()

    def log(self, *, run_id: str, experiment: str, config: str, turn: int | None,
            direction: str, type_: str, meta: dict | None = None) -> int:
        rec = {
            "run_id": run_id,
            "experiment": experiment,
            "config": config,
            "turn": turn,
            "t_ns": mono_ns(),
            "dir": direction,
            "type": type_,
            "meta": meta or {},
        }
        with self._lock:
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec["t_ns"]

    def close(self):
        self._fh.close()


def existing_run_ids(experiment: str) -> set[str]:
    """Run ids already present in the raw log (used by --resume)."""
    path = RAW_DIR / f"{experiment}.jsonl"
    ids: set[str] = set()
    if not path.exists():
        return ids
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["run_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def read_events(experiment: str) -> list[dict]:
    path = RAW_DIR / f"{experiment}.jsonl"
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def unb64(s: str) -> bytes:
    return base64.b64decode(s)


class FramePacer:
    """Real-time pacer, design.md 7.2.

    The k-th frame is due at start + k * frame_s. Drift-corrected: we sleep
    until each absolute deadline instead of sleeping frame_s in a loop.
    """

    def __init__(self, frame_ms: int = 20):
        self.frame_s = frame_ms / 1000.0

    async def run(self, frames, on_frame, should_continue=None, max_tail_s: float = 3.0):
        """Send/schedule frames on a drift-corrected schedule.

        frames: iterable of PCM bytes (one entry per frame) for the speech part.
        on_frame: async callable(frame_bytes: bytes|None, index: int, t_ns: int).
                  Called with None to mean "silence frame" (tail) and with
                  bytes for speech frames.
        should_continue: callable() -> bool checked during the silence tail.
        Returns (t_eos, n_frames_sent).
        """
        loop = asyncio.get_running_loop()
        start = loop.time()
        k = 0
        t_eos = None

        async def tick(frame):
            nonlocal k
            target = start + k * self.frame_s
            now = loop.time()
            if target > now:
                await asyncio.sleep(target - now)
            t_ns = mono_ns()
            await on_frame(frame, k, t_ns)
            k += 1
            return t_ns

        # 300 ms silence pre-roll
        for _ in range(int(0.300 / self.frame_s)):
            await tick(None)
        # speech
        for frame in frames:
            t_eos = await tick(frame)
        # silence tail until the transcript is final or max_tail_s pass
        tail_end = loop.time() + max_tail_s
        while loop.time() < tail_end:
            if should_continue is not None and not should_continue():
                break
            await tick(None)
        return t_eos, k


def silence_frame(frame_ms: int, sample_rate: int) -> bytes:
    return b"\x00\x00" * int(sample_rate * frame_ms / 1000)


def chunk_pcm(pcm: bytes, sample_rate: int, frame_ms: int) -> list[bytes]:
    """Split PCM s16le mono bytes into frame_ms-sized frames."""
    n = int(sample_rate * frame_ms / 1000) * 2
    frames = [pcm[i:i + n] for i in range(0, len(pcm), n)]
    if frames and len(frames[-1]) < n:
        frames[-1] = frames[-1].ljust(n, b"\x00")
    return frames


def save_pcm(path: Path, pcm: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pcm)


def git_sha() -> str:
    import subprocess
    try:
        root = Path(__file__).resolve().parent.parent
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root).decode().strip()
    except Exception:
        return ""
