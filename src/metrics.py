"""Measurement primitives: peak memory + a timing context for TTFT/ITL capture."""
import time
import threading
from contextlib import contextmanager

import psutil

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


class MemoryTracker:
    """Polls RSS (and MPS/accelerator if available) at intervals; reports peak MB."""

    def __init__(self, interval=0.05):
        self.interval = interval
        self._stop = threading.Event()
        self._peak_rss_mb = 0.0
        self._thread = None

    def _loop(self):
        proc = psutil.Process()
        while not self._stop.is_set():
            rss = proc.memory_info().rss / (1024 * 1024)
            if rss > self._peak_rss_mb:
                self._peak_rss_mb = rss
            time.sleep(self.interval)

    def start(self):
        self._stop.clear()
        self._peak_rss_mb = 0.0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        return self._peak_rss_mb

    @property
    def peak_rss_mb(self) -> float:
        return self._peak_rss_mb


@contextmanager
def peak_memory(interval: float = 0.05):
    """Yields a MemoryTracker; read .peak_rss_mb after the block."""
    mt = MemoryTracker(interval)
    mt.start()
    try:
        yield mt
    finally:
        mt.stop()


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def summarize_token_times(t_ms: list) -> dict:
    """Given cumulative per-token timestamps, return TTFT, mean ITL, throughput."""
    if not t_ms:
        return {"ttft_ms": None, "itl_mean_ms": None, "itl_p50_ms": None,
                "throughput_tps": None, "n_tokens": 0}
    ttft = t_ms[0]
    itls = [t_ms[i] - t_ms[i - 1] for i in range(1, len(t_ms))] if len(t_ms) > 1 else []
    mean_itl = sum(itls) / len(itls) if itls else None
    p50 = sorted(itls)[len(itls) // 2] if itls else None
    total = t_ms[-1] - t_ms[0]
    tps = (len(t_ms) / total * 1000.0) if (total > 0 and len(t_ms) > 1) else None
    return {"ttft_ms": ttft, "itl_mean_ms": mean_itl, "itl_p50_ms": p50,
            "throughput_tps": tps, "n_tokens": len(t_ms)}
