"""Tests for src/analyze_perf.py — loader, write_table, and plot functions."""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.analyze_perf import load_results, plot_path_comparison, plot_quant_sweep, write_table
from src.report import format_record_md

_SAMPLE_REC = {
    "scenario": "airllm", "size": "14B", "quant": "fp16", "model": "",
    "ok": True, "wall_ms": 63000.0, "load_ms": 1200.0,
    "peak_rss_mb": 4500.0, "estimated_kwh": 0.7, "repeat": 0,
    "timing": {"ttft_ms": 14000.0, "itl_mean_ms": 12800.0,
               "itl_p50_ms": 12900.0, "throughput_tps": 0.08, "n_tokens": 49},
    "tokens": {"input_tokens": 16, "output_tokens": 48, "total_tokens": 64},
    "quality": {"on_topic": True, "coherent": True, "term_hits": 3},
    "output": "Virtual memory paging works by…",
}
_AIRLLM_ROW = {
    "scenario": "airllm", "size": "14B", "quant": "fp16", "ok": True,
    "ttft_ms": 14000.0, "itl_mean_ms": 12800.0, "throughput_tps": 0.08,
    "peak_rss_mb": 4500.0, "wall_ms": 630000.0, "estimated_kwh": 7.0,
    "model": "", "error": "",
}
_OLLAMA_Q4 = {
    "scenario": "ollama", "size": "14B", "quant": "q4", "ok": True,
    "ttft_ms": 8400.0, "itl_mean_ms": 103.0, "throughput_tps": 9.9,
    "peak_rss_mb": 260.0, "wall_ms": 13000.0, "estimated_kwh": 0.15,
    "model": "qwen2.5:14b", "error": "",
}
_OLLAMA_Q8_FAIL = {
    "scenario": "ollama", "size": "14B", "quant": "q8", "ok": False,
    "ttft_ms": None, "itl_mean_ms": None, "throughput_tps": None,
    "peak_rss_mb": None, "wall_ms": 720000.0, "estimated_kwh": None,
    "model": "qwen2.5:14b-instruct-q8_0", "error": "timeout",
}


def _make_df():
    return pd.DataFrame([_AIRLLM_ROW, _OLLAMA_Q4, _OLLAMA_Q8_FAIL])


class TestLoadResults:
    def test_loads_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "airllm_14B__1.md").write_text(format_record_md(_SAMPLE_REC))
            with patch("src.analyze_perf.config") as mc:
                mc.RESULTS_DIR = tmp
                mc.SUBJECT = "14B"
                df = load_results()
        assert len(df) == 1 and df.iloc[0]["scenario"] == "airllm"

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.analyze_perf.config") as mc:
                mc.RESULTS_DIR = tmp
                mc.SUBJECT = "14B"
                assert load_results().empty

    def test_normalizes_unavailable_numeric_values(self):
        failed = {
            "scenario": "baseline", "size": "14B", "quant": "fp16",
            "ok": False, "wall_ms": 1000.0, "peak_rss_mb": "—",
            "estimated_kwh": "—", "timing": {"ttft_ms": "—"},
        }
        with patch("src.analyze_perf.report.load_results_md", return_value=[failed]):
            df = load_results()
        assert pd.isna(df.iloc[0]["peak_rss_mb"])
        assert pd.isna(df.iloc[0]["ttft_ms"])


class TestWriteTable:
    def test_write_calls_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.analyze_perf.config") as mc, \
                 patch("src.analyze_perf.report") as mr:
                mc.RESULTS_DIR = Path(tmp)
                write_table(_make_df())
            mr.write_summary.assert_called_once()


class TestPlotPathComparison:
    def test_runs_without_error(self):
        fig, ax1, ax2 = MagicMock(), MagicMock(), MagicMock()
        with patch("src.analyze_perf.plt") as mp, \
             patch("src.analyze_perf.config") as mc:
            mp.subplots.return_value = (fig, (ax1, ax2))
            mc.SUBJECT = "14B"
            mc.FIGURES_DIR = Path(tempfile.mkdtemp())
            plot_path_comparison(_make_df())
        mp.subplots.assert_called_once()

    def test_empty_df(self):
        fig, ax1, ax2 = MagicMock(), MagicMock(), MagicMock()
        with patch("src.analyze_perf.plt") as mp, \
             patch("src.analyze_perf.config") as mc:
            mp.subplots.return_value = (fig, (ax1, ax2))
            mc.SUBJECT = "14B"
            mc.FIGURES_DIR = Path(tempfile.mkdtemp())
            plot_path_comparison(pd.DataFrame(columns=_make_df().columns))


class TestPlotQuantSweep:
    def test_runs_with_data(self):
        axes = [MagicMock(), MagicMock(), MagicMock()]
        with patch("src.analyze_perf.plt") as mp, \
             patch("src.analyze_perf.config") as mc:
            mp.subplots.return_value = (MagicMock(), axes)
            mc.FIGURES_DIR = Path(tempfile.mkdtemp())
            plot_quant_sweep(_make_df())

    def test_skips_no_ollama_rows(self):
        with patch("src.analyze_perf.plt") as mp, \
             patch("src.analyze_perf.config") as mc:
            mc.FIGURES_DIR = Path(tempfile.mkdtemp())
            plot_quant_sweep(pd.DataFrame([_AIRLLM_ROW]))
        mp.subplots.assert_not_called()
