"""Tests for bench_driver, run_sweep, experiments/prompts, and generate_report."""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from experiments.prompts import SCENARIOS, score_output
from src import config as real_config
from src.bench_driver import _annotate, _save, cmd_ollama
from src.generate_report import generate
from src.run_sweep import run_all


class TestScoreOutput:
    def test_empty(self):
        r = score_output("")
        assert r["on_topic"] is False and r["term_hits"] == 0

    def test_rich_output(self):
        out = ("Virtual memory paging splits the address space into fixed-size pages "
               "stored on disk; the OS swaps pages in and out of physical memory on demand.")
        r = score_output(out)
        assert r["on_topic"] is True and r["term_hits"] >= 2

    def test_complete_ends_period(self):
        assert score_output("Virtual memory uses paging to manage physical memory.")["complete"]

    def test_incomplete_truncated(self):
        assert not score_output("Virtual memory paging works by dividing memory into")["complete"]

    def test_scenarios_keys(self):
        for sc in SCENARIOS.values():
            assert "prompt" in sc and "max_new_tokens" in sc


class TestAnnotate:
    def test_adds_kwh(self):
        r = _annotate({"wall_ms": 3_600_000.0, "output": "paging answer"})
        assert r["estimated_kwh"] > 0

    def test_adds_quality(self):
        r = _annotate({"wall_ms": 1000.0, "output": "paging virtual memory answer"})
        assert isinstance(r["quality"], dict)

    def test_missing_output(self):
        assert "quality" in _annotate({"wall_ms": 1000.0})


class TestBenchSave:
    def test_writes_file(self):
        rec = {"scenario": "airllm", "size": "14B", "ok": True, "wall_ms": 1000.0}
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.bench_driver.config") as mc, \
                 patch("src.bench_driver.time.time", return_value=9999999):
                mc.RESULTS_DIR = Path(tmp)
                _save(rec, "airllm_14B_fp16")
            files = list(Path(tmp).glob("*.md"))
        assert len(files) == 1 and "airllm_14B_fp16" in files[0].name


class TestCmdOllama:
    def test_calls_run_and_save(self):
        args = MagicMock()
        args.size, args.quants, args.repeats = "14b", ["q4"], 1
        fake_rec = {
            "scenario": "ollama", "ok": True, "wall_ms": 1000.0,
            "output": "paging answer",
            "timing": {"ttft_ms": 100.0, "itl_mean_ms": 80.0,
                       "throughput_tps": 12.0, "itl_p50_ms": 80.0, "n_tokens": 10},
        }
        with patch("src.bench_driver.run_ollama.run", return_value=fake_rec) as mr, \
             patch("src.bench_driver._save") as ms, \
             patch("src.bench_driver.config") as mc:
            mc.HW_LOAD_W = 40
            cmd_ollama(args)
        mr.assert_called_once()
        ms.assert_called_once()


class TestRunSweep:
    def test_calls_airllm_per_size(self):
        fake = {"scenario": "airllm", "size": "0.5B", "ok": True, "wall_ms": 1000.0}
        with patch("src.run_sweep.run_airllm.run", return_value=fake) as mr, \
             patch("src.run_sweep.config") as mc:
            mc.SWEEP_SIZES = ["0.5B", "1.5B"]
            mc.MODELS = real_config.MODELS
            results = run_all("explain paging", 24, sizes=["0.5B", "1.5B"])
        assert mr.call_count == 2
        assert all("layers" in r and "params_b" in r for r in results)

    def test_uses_default_sizes(self):
        fake = {"scenario": "airllm", "ok": True, "wall_ms": 1000.0}
        with patch("src.run_sweep.run_airllm.run", return_value=dict(fake)), \
             patch("src.run_sweep.config") as mc:
            mc.SWEEP_SIZES = ["0.5B"]
            mc.MODELS = real_config.MODELS
            assert len(run_all("hi", 10)) == 1


class TestGenerateReport:
    def test_generates_markdown(self):
        md = generate()
        assert "# EX05" in md and "## 1. Headline comparison" in md

    def test_contains_figures(self):
        md = generate()
        assert "size_scaling.png" in md and "path_comparison.png" in md
