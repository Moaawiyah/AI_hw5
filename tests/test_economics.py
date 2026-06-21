"""Tests for src/economics.py and src/config.py."""
import numpy as np
import pytest

from src import config
from src.economics import (
    api_cost_per_request,
    cloud_gpu_cost_per_request,
    cumulative,
    find_breakeven,
    onprem_cost_per_request,
)


class TestApiCost:
    def test_positive(self):
        assert api_cost_per_request("hello world", "paging answer") > 0

    def test_cache_reduces_cost(self):
        full = api_cost_per_request("hello world", "paging answer", cache_fraction=0.0)
        cached = api_cost_per_request("hello world", "paging answer", cache_fraction=0.8)
        assert cached < full

    def test_empty_strings(self):
        assert api_cost_per_request("", "") == 0.0


class TestOnpremCost:
    def test_positive(self):
        assert onprem_cost_per_request(seconds=15.0, monthly_volume=1000) > 0

    def test_higher_volume_lowers_per_req(self):
        low = onprem_cost_per_request(15.0, 100)
        high = onprem_cost_per_request(15.0, 10_000)
        assert high < low


class TestCloudGpuCost:
    def test_proportional(self):
        c1 = cloud_gpu_cost_per_request(3600.0)
        c2 = cloud_gpu_cost_per_request(7200.0)
        assert pytest.approx(c2) == 2 * c1


class TestCumulativeAndBreakeven:
    def test_structure(self):
        vols = np.array([10.0, 100.0, 1000.0])
        r = cumulative(vols, "hello", "world", onprem_seconds=15)
        assert set(r.keys()) >= {"volumes", "api", "api_cached", "onprem", "cloud_gpu"}
        assert len(r["api"]) == 3

    def test_find_breakeven_found(self):
        a = np.array([10.0, 5.0, 1.0])
        b = np.array([8.0, 3.0, 0.5])
        be = find_breakeven(a, b, np.array([1.0, 2.0, 3.0]))
        assert be == pytest.approx(1.0)

    def test_find_breakeven_never(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0, 20.0, 30.0])
        assert find_breakeven(a, b, np.array([1.0, 2.0, 3.0])) is None


class TestConfig:
    def test_models_registered(self):
        for key in ("0.5B", "1.5B", "14B"):
            assert key in config.MODELS

    def test_model_fields(self):
        for meta in config.MODELS.values():
            assert {"id", "params_b", "layers"} <= set(meta)

    def test_dirs_exist(self):
        assert config.ROOT.exists()
        assert config.RESULTS_DIR.exists()
        assert config.FIGURES_DIR.exists()

    def test_subject_in_models(self):
        assert config.SUBJECT in config.MODELS

    def test_api_prices(self):
        for prices in config.API_PRICES.values():
            assert "input" in prices and "output" in prices
