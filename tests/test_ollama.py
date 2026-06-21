"""Tests for src/run_ollama.py — model name mapping, streaming, and run()."""
import json
from unittest.mock import MagicMock, patch

from src.run_ollama import _model_name, _stream, run as ollama_run


class TestModelName:
    def test_q4_default(self):
        assert _model_name("14b", "q4") == "qwen2.5:14b"

    def test_q2_suffix(self):
        name = _model_name("14b", "q2")
        assert name.startswith("qwen2.5:") and ("q2" in name or "instruct" in name)

    def test_q8_suffix(self):
        name = _model_name("14b", "q8")
        assert "q8" in name.lower() or "instruct" in name.lower()

    def test_unknown_quant_uses_default(self):
        assert _model_name("14b", "unknown") == "qwen2.5:14b"

    def test_size_lowercased(self):
        assert "14b" in _model_name("14B", "q4")


def _fake_resp(chunks):
    """Build a fake urllib streaming response from a list of chunk dicts."""
    lines = [json.dumps(c).encode() + b"\n" for c in chunks]
    resp = MagicMock()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    resp.__iter__ = lambda s: iter(lines)
    return resp


class TestStream:
    def test_yields_tokens(self):
        resp = _fake_resp([
            {"response": "Virtual ", "done": False},
            {"response": "memory ", "done": False},
            {"response": "paging.", "done": True},
        ])
        with patch("src.run_ollama.urllib.request.urlopen", return_value=resp):
            assert list(_stream("qwen2.5:14b", "explain", 10)) == [
                "Virtual ", "memory ", "paging."
            ]

    def test_stops_at_done(self):
        resp = _fake_resp([
            {"response": "hello", "done": False},
            {"done": True},
            {"response": "ignored", "done": False},
        ])
        with patch("src.run_ollama.urllib.request.urlopen", return_value=resp):
            assert "ignored" not in list(_stream("qwen2.5:14b", "hi", 5))

    def test_skips_empty_response(self):
        resp = _fake_resp([{"response": "", "done": False}, {"response": "ok", "done": True}])
        with patch("src.run_ollama.urllib.request.urlopen", return_value=resp):
            assert list(_stream("qwen2.5:14b", "hi", 5)) == ["ok"]


def _ping_then_stream(chunks):
    ping = MagicMock()
    ping.close = MagicMock()
    return MagicMock(side_effect=[ping, _fake_resp(chunks)])


class TestOllamaRun:
    def test_run_success(self):
        side = _ping_then_stream([{"response": "Paging works.", "done": False}, {"done": True}])
        with patch("src.run_ollama.urllib.request.urlopen", side):
            rec = ollama_run("14b", "explain paging", 10, tag="q4")
        assert rec["ok"] is True
        assert "Paging works." in rec["output"]
        assert "timing" in rec

    def test_run_ping_failure(self):
        with patch("src.run_ollama.urllib.request.urlopen",
                   side_effect=ConnectionRefusedError("offline")):
            rec = ollama_run("14b", "explain paging", 10, tag="q4")
        assert rec["ok"] is False and "error" in rec

    def test_run_stream_error(self):
        ping = MagicMock()
        ping.close = MagicMock()
        with patch("src.run_ollama.urllib.request.urlopen",
                   MagicMock(side_effect=[ping, RuntimeError("broken")])):
            rec = ollama_run("14b", "explain paging", 10, tag="q4")
        assert rec["ok"] is False
