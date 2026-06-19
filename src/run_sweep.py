"""Original extension (§5.7): model-size scaling sweep.

Runs AirLLM (FP16) across 0.5B/7B/14B/32B at a fixed small token budget to chart
how TTFT and per-token decode latency scale with layer count. Reuses run_airllm.
"""
from src import config
from src import run_airllm


def run_all(prompt: str, max_new_tokens: int, sizes=None) -> list:
    """Run the size sweep. Returns a list of result dicts."""
    sizes = sizes or config.SWEEP_SIZES
    results = []
    for size in sizes:
        print(f"[sweep] AirLLM FP16 {size} ...")
        rec = run_airllm.run(size, "fp16", prompt, max_new_tokens)
        rec["scenario"] = "size_sweep"
        rec["layers"] = config.MODELS[size]["layers"]
        rec["params_b"] = config.MODELS[size]["params_b"]
        results.append(rec)
    return results
