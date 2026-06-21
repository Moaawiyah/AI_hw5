"""Tests for src/hf_utils.py — path helpers, untie_embeddings, download mocks."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src import config as real_config
from src import config
from src.hf_utils import (
    download_family,
    download_model,
    get_token,
    local_dir,
    untie_embeddings,
)


class TestHfUtils:
    def test_local_dir_structure(self):
        p = local_dir("Qwen/Qwen2.5-14B-Instruct")
        assert p.name == "Qwen2.5-14B-Instruct"
        assert p.parent == config.MODELS_CACHE

    def test_get_token_returns_string(self):
        assert isinstance(get_token(), str)

    def test_get_token_from_env(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "test-token-123")
        assert get_token() == "test-token-123"

    def test_download_model_unknown_key(self):
        with pytest.raises(KeyError):
            download_model("999B")


class TestUntieEmbeddings:
    def test_skips_if_no_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            untie_embeddings(tmp)  # no config.json → no-op

    def test_skips_if_not_tied(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.json").write_text(
                json.dumps({"tie_word_embeddings": False})
            )
            (Path(tmp) / "model.safetensors").write_bytes(b"fake")
            untie_embeddings(tmp)  # tie_word_embeddings=False → no-op

    def test_skips_if_no_safetensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.json").write_text(
                json.dumps({"tie_word_embeddings": True})
            )
            untie_embeddings(tmp)  # model.safetensors absent → no-op


class TestDownloadModel:
    def test_calls_snapshot_download(self):
        with patch("src.hf_utils.snapshot_download") as mock_dl, \
             patch("src.hf_utils.config") as mc:
            mc.MODELS = real_config.MODELS
            mc.MODELS_CACHE = Path(tempfile.mkdtemp())
            download_model("14B")
        mock_dl.assert_called_once()
        assert mock_dl.call_args[1]["repo_id"] == "Qwen/Qwen2.5-14B-Instruct"

    def test_download_family_calls_each(self):
        with patch("src.hf_utils.download_model") as mock_dm, \
             patch("src.hf_utils.config") as mc:
            mc.SWEEP_SIZES = ["0.5B", "1.5B"]
            mc.MODELS = real_config.MODELS
            mock_dm.side_effect = lambda s: Path(f"/fake/{s}")
            result = download_family(["0.5B", "1.5B"])
        assert mock_dm.call_count == 2 and "0.5B" in result
