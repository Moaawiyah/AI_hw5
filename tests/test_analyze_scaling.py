"""Tests for src/analyze_scaling.py — _load_sweep branches and plot()."""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from src import config as real_config
from src.analyze_scaling import _load_sweep, plot as scaling_plot
from src.report import format_record_md

_REC = {
    "scenario": "airllm", "size": "14B", "quant": "fp16",
    "ok": True, "wall_ms": 63000.0, "load_ms": 1200.0,
    "peak_rss_mb": 4500.0, "estimated_kwh": 0.7, "repeat": 0,
    "timing": {"ttft_ms": 14000.0, "itl_mean_ms": 12800.0,
               "itl_p50_ms": 12900.0, "throughput_tps": 0.08, "n_tokens": 49},
    "tokens": {"input_tokens": 16, "output_tokens": 48, "total_tokens": 64},
    "quality": {"on_topic": True, "coherent": True, "term_hits": 3},
    "output": "Virtual memory paging works by…",
}


def _write(tmp, rec):
    name = f"airllm_{rec.get('size', 'X')}_{rec.get('quant', 'fp16')}__1.md"
    (Path(tmp) / name).write_text(format_record_md(rec))


class TestLoadSweep:
    def test_loads_fp16(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, _REC)
            with patch("src.analyze_scaling.config") as mc:
                mc.RESULTS_DIR = tmp
                mc.MODELS = real_config.MODELS
                pts = _load_sweep()
        assert len(pts) == 1 and pts[0][0] == "14B"

    def test_skips_non_fp16(self):
        rec = {**_REC, "quant": "q4"}
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, rec)
            with patch("src.analyze_scaling.config") as mc:
                mc.RESULTS_DIR = tmp
                mc.MODELS = real_config.MODELS
                assert _load_sweep() == []

    def test_skips_ok_false(self):
        rec = {**_REC, "ok": False}
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, rec)
            with patch("src.analyze_scaling.config") as mc:
                mc.RESULTS_DIR = tmp
                mc.MODELS = real_config.MODELS
                assert _load_sweep() == []

    def test_skips_unknown_size(self):
        rec = {**_REC, "size": "999B"}
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, rec)
            with patch("src.analyze_scaling.config") as mc:
                mc.RESULTS_DIR = tmp
                mc.MODELS = real_config.MODELS
                assert _load_sweep() == []

    def test_skips_missing_itl(self):
        rec = {**_REC, "timing": {"ttft_ms": 100.0, "n_tokens": 5}}
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, rec)
            with patch("src.analyze_scaling.config") as mc:
                mc.RESULTS_DIR = tmp
                mc.MODELS = real_config.MODELS
                assert _load_sweep() == []


class TestScalingPlot:
    def test_plot_with_data(self):
        ax1, ax2 = MagicMock(), MagicMock()
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, _REC)
            with patch("src.analyze_scaling.config") as mc, \
                 patch("src.analyze_scaling.plt") as mp:
                mc.RESULTS_DIR = tmp
                mc.MODELS = real_config.MODELS
                mc.FIGURES_DIR = Path(tmp)
                mp.subplots.return_value = (MagicMock(), [ax1, ax2])
                scaling_plot()
        mp.subplots.assert_called_once()

    def test_plot_no_data_skips(self, capsys):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.analyze_scaling.config") as mc, \
                 patch("src.analyze_scaling.plt"):
                mc.RESULTS_DIR = tmp
                mc.MODELS = real_config.MODELS
                scaling_plot()
        assert "skip" in capsys.readouterr().out
