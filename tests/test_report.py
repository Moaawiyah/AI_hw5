"""Tests for src/report.py and src/token_count.py."""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.report import format_record_md, format_summary_md, load_results_md, parse_record_md
from src.token_count import count_tokens, exact_tokens_with_tokenizer, request_tokens

SAMPLE_REC = {
    "scenario": "airllm", "size": "14B", "quant": "fp16", "model": "",
    "ok": True, "wall_ms": 63000.0, "load_ms": 1200.0,
    "peak_rss_mb": 4500.0, "estimated_kwh": 0.7, "repeat": 0,
    "timing": {"ttft_ms": 14000.0, "itl_mean_ms": 12800.0,
               "itl_p50_ms": 12900.0, "throughput_tps": 0.08, "n_tokens": 49},
    "tokens": {"input_tokens": 16, "output_tokens": 48, "total_tokens": 64},
    "quality": {"on_topic": True, "coherent": True, "term_hits": 3},
    "output": "Virtual memory paging works by…",
}


class TestCountTokens:
    def test_empty(self):
        assert count_tokens("") == 0

    def test_nonempty(self):
        assert count_tokens("hello world") > 0

    def test_longer_is_more(self):
        assert count_tokens("hi " * 50) > count_tokens("hi")


class TestRequestTokens:
    def test_structure(self):
        r = request_tokens("prompt", "output")
        assert set(r) == {"input_tokens", "output_tokens", "total_tokens"}
        assert r["total_tokens"] == r["input_tokens"] + r["output_tokens"]

    def test_empty_strings(self):
        assert request_tokens("", "")["total_tokens"] == 0


class TestExactTokensWithTokenizer:
    def test_with_mock_tokenizer(self):
        tok = MagicMock()
        tok.side_effect = lambda text: {"input_ids": list(range(len(text.split())))}
        r = exact_tokens_with_tokenizer(tok, "hello world", "foo bar baz")
        assert r["input_tokens"] == 2
        assert r["output_tokens"] == 3

    def test_fallback_on_error(self):
        tok = MagicMock(side_effect=RuntimeError("boom"))
        assert "input_tokens" in exact_tokens_with_tokenizer(tok, "hello", "world")


class TestFormatAndParse:
    def test_roundtrip(self):
        parsed = parse_record_md(format_record_md(SAMPLE_REC))
        assert parsed["scenario"] == "airllm"
        assert parsed["ok"] is True
        assert parsed["timing"]["ttft_ms"] == pytest.approx(14000.0)

    def test_failed_record(self):
        md = format_record_md({"scenario": "baseline", "ok": False,
                                "error": "OOM", "wall_ms": 5000.0})
        assert "failed" in md and "OOM" in md

    def test_format_summary_md(self):
        out = format_summary_md([SAMPLE_REC])
        assert "airllm" in out and "14B" in out and "|" in out

    def test_format_summary_renders_nan_as_unavailable(self):
        out = format_summary_md([{"scenario": "baseline", "ttft_ms": float("nan")}])
        assert "nan" not in out.lower()

    def test_parse_coerce_types(self):
        parsed = parse_record_md(format_record_md(SAMPLE_REC))
        assert isinstance(parsed["ok"], bool)
        assert isinstance(parsed["wall_ms"], float)
        assert isinstance(parsed["repeat"], int)


class TestLoadResultsMd:
    def test_loads_from_tempdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "airllm_14B_fp16__999.md").write_text(format_record_md(SAMPLE_REC))
            results = load_results_md(tmp)
        assert len(results) == 1 and results[0]["scenario"] == "airllm"

    def test_skips_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "summary.md").write_text("# summary\n| a | b |\n")
            assert load_results_md(tmp) == []

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert load_results_md(tmp) == []
