"""Cross-platform accelerator selection for the direct baseline."""
from unittest.mock import patch

from src.run_baseline import _device


def test_prefers_cuda():
    with patch("src.run_baseline.torch.cuda.is_available", return_value=True):
        assert _device() == "cuda"


def test_uses_mps_when_cuda_is_unavailable():
    with patch("src.run_baseline.torch.cuda.is_available", return_value=False), \
         patch("src.run_baseline.torch.backends.mps.is_available", return_value=True):
        assert _device() == "mps"


def test_falls_back_to_cpu():
    with patch("src.run_baseline.torch.cuda.is_available", return_value=False), \
         patch("src.run_baseline.torch.backends.mps.is_available", return_value=False):
        assert _device() == "cpu"
