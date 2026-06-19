"""Economic analysis (§5.5): API cost vs On-Prem (CAPEX+OPEX) vs optional Cloud GPU.
All assumptions live in config.py and must be stated in the report.
"""
import numpy as np

from src import config
from src.token_count import count_tokens


def api_cost_per_request(prompt: str, output: str, provider="gpt-4o",
                         cache_fraction=0.0) -> float:
    """USD cost for one request. cache_fraction is the share of input tokens that
    hit prompt/context-cache pricing (PageAttention-style, §5.5)."""
    price = config.API_PRICES[provider]
    tin, tout = count_tokens(prompt), count_tokens(output)
    cached = tin * cache_fraction
    fresh = tin - cached
    return (fresh * price["input"] + cached * price["cache_input"]
            + tout * price["output"]) / 1e6


def onprem_cost_per_request(seconds: float, monthly_volume: int) -> float:
    """Amortized CAPEX + OPEX (electricity, maintenance) per request."""
    monthly_requests = monthly_volume
    months_life = config.HW_LIFETIME_YEARS * 12
    capex_per_req = config.HW_CAPEX_USD / (months_life * monthly_requests)
    kwh = (config.HW_LOAD_W / 1000.0) * (seconds / 3600.0)
    elec_per_req = kwh * config.ELECTRICITY_KWH_USD
    maint_per_req = (config.HW_CAPEX_USD * config.MAINT_FRAC_PER_YEAR / 12) / monthly_requests
    return capex_per_req + elec_per_req + maint_per_req


def cloud_gpu_cost_per_request(seconds: float) -> float:
    return config.CLOUD_GPU_USD_PER_HOUR * (seconds / 3600.0)


def cumulative(volumes, prompt, output, onprem_seconds):
    """Return dict of arrays: cumulative cost vs monthly volume for each option."""
    api = np.array([api_cost_per_request(prompt, output) * v for v in volumes])
    api_cached = np.array([api_cost_per_request(prompt, output, cache_fraction=0.8) * v
                           for v in volumes])
    onp = np.array([onprem_cost_per_request(onprem_seconds, v) * v for v in volumes])
    cloud = np.array([cloud_gpu_cost_per_request(onprem_seconds) * v for v in volumes])
    return {"volumes": volumes, "api": api, "api_cached": api_cached,
            "onprem": onp, "cloud_gpu": cloud}


def find_breakeven(series_a, series_b, volumes):
    """First volume where series_b becomes cheaper than series_a."""
    a = np.asarray(series_a)
    b = np.asarray(series_b)
    cheaper = np.where(b < a)[0]
    return volumes[cheaper[0]] if len(cheaper) else None


if __name__ == "__main__":
    from experiments.prompts import PRIMARY_PROMPT
    sample_out = ("Paging moves unused memory pages to disk and back as needed, "
                  "letting programs use more memory than physically present.")
    # Realistic on-prem latency = Ollama Q4 path (~15s for ~50 tokens at 9.7 tok/s
    # + load). Note: AirLLM's ~600s/request makes its electricity cost alone exceed
    # the API cost, so it never breaks even on pure cost (see report).
    vols = np.logspace(1, 6, 60)
    for secs, label in [(15, "Ollama Q4 (~15s/req)"), (600, "AirLLM (~600s/req)")]:
        c = cumulative(vols, PRIMARY_PROMPT, sample_out, onprem_seconds=secs)
        be = find_breakeven(c["api"], c["onprem"], vols)
        print(f"{label}: On-Prem cheaper than API at "
              f"{('%.0f' % be) if be else 'NEVER (on-prem electricity > API)'} req/month")
