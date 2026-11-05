"""Playback simulator (design.md 7.4).

Computes stalls from chunk arrival times, as if playback started at the first
chunk. Also reports how much audio is buffered ahead of the playhead.
"""
from __future__ import annotations


class PlaybackSimulator:
    def __init__(self, sample_rate: int = 24000, jitter_ms: float = 20.0):
        self.bytes_per_ms = sample_rate * 2 / 1000.0  # 16-bit mono PCM
        self.jitter_ms = jitter_ms
        self.chunks: list[tuple[int, int]] = []  # (t_ns, n_bytes)
        self.stalls = 0
        self._play_end_ns: float | None = None  # ns playhead end if no underrun
        self._t_first_ns: int | None = None

    def dur_ms(self, n_bytes: int) -> float:
        return n_bytes / self.bytes_per_ms

    def feed(self, t_ns: int, n_bytes: int):
        self.chunks.append((t_ns, n_bytes))
        dur_ms = self.dur_ms(n_bytes)
        if self._play_end_ns is None:
            self._t_first_ns = t_ns
            self._play_end_ns = t_ns + dur_ms * 1e6
        elif t_ns > self._play_end_ns + self.jitter_ms * 1e6:
            self.stalls += 1
            self._play_end_ns = t_ns + dur_ms * 1e6
        else:
            self._play_end_ns += dur_ms * 1e6

    def buffered_ahead_ms(self, t_ns: int) -> float:
        """Audio delivered but not yet played at t_ns."""
        if self._play_end_ns is None or len(self.chunks) == 0:
            return 0.0
        # playhead at t_ns: started at t_first, has consumed t - t_first of audio
        consumed_ns = t_ns - self._t_first_ns
        ahead = self._play_end_ns - self._t_first_ns - consumed_ns
        return max(ahead, 0.0) / 1e6

    def total_audio_ms(self) -> float:
        return sum(self.dur_ms(n) for _, n in self.chunks)
