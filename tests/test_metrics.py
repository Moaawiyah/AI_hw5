"""Tests for src/metrics.py — MemoryTracker, summarize_token_times, power math."""
import time

import pytest

from src.metrics import (
    MemoryTracker,
    estimate_power_kwh,
    now_ms,
    peak_memory,
    summarize_token_times,
)


class TestSummarizeTokenTimes:
    def test_empty(self):
        r = summarize_token_times([])
        assert r["n_tokens"] == 0
        assert r["ttft_ms"] is None

    def test_single_token(self):
        r = summarize_token_times([100.0])
        assert r["ttft_ms"] == 100.0
        assert r["itl_mean_ms"] is None
        assert r["n_tokens"] == 1

    def test_multiple_tokens(self):
        r = summarize_token_times([100.0, 200.0, 350.0])
        assert r["ttft_ms"] == 100.0
        assert r["itl_mean_ms"] == pytest.approx(125.0)
        assert r["n_tokens"] == 3
        assert r["throughput_tps"] is not None and r["throughput_tps"] > 0

    def test_two_tokens_throughput(self):
        r = summarize_token_times([0.0, 1000.0])
        assert r["throughput_tps"] == pytest.approx(2.0)

    def test_p50(self):
        r = summarize_token_times([0.0, 100.0, 300.0, 600.0])
        itls = [100.0, 200.0, 300.0]
        p50 = sorted(itls)[len(itls) // 2]
        assert r["itl_p50_ms"] == pytest.approx(p50)


class TestEstimatePowerKwh:
    def test_zero(self):
        assert estimate_power_kwh(0, 40) == 0.0

    def test_one_hour(self):
        assert estimate_power_kwh(3_600_000, 40) == pytest.approx(0.04)

    def test_half_hour(self):
        assert estimate_power_kwh(1_800_000, 100) == pytest.approx(0.05)


class TestNowMs:
    def test_monotone(self):
        t1 = now_ms()
        time.sleep(0.01)
        t2 = now_ms()
        assert t2 > t1

    def test_unit(self):
        assert now_ms() > 1_000_000


class TestMemoryTracker:
    def test_peak_increases(self):
        mt = MemoryTracker(interval=0.01)
        mt.start()
        time.sleep(0.05)
        assert mt.stop() > 0

    def test_peak_rss_property(self):
        mt = MemoryTracker(interval=0.01)
        mt.start()
        time.sleep(0.02)
        mt.stop()
        assert mt.peak_rss_mb > 0


class TestPeakMemoryContext:
    def test_context_manager(self):
        with peak_memory(interval=0.01) as mt:
            _ = [0] * 100_000
        assert mt.peak_rss_mb > 0
