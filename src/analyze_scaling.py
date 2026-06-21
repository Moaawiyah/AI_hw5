"""Original-extension figure (§5.7): decode latency vs model size (log-log)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config, report


def _load_sweep():
    """Read AirLLM FP16 runs across sizes for the scaling extension (§5.7)."""
    pts = []
    for r in report.load_results_md(config.RESULTS_DIR):
        if r.get("scenario") != "airllm" or r.get("quant") != "fp16":
            continue
        if not r.get("ok") or r.get("size") not in config.MODELS:
            continue
        t = r.get("timing", {}) or {}
        if t.get("itl_mean_ms"):
            meta = config.MODELS[r["size"]]
            pts.append((r["size"], meta["params_b"], meta["layers"],
                        t["itl_mean_ms"], t.get("ttft_ms")))
    return pts


def _save(fig, name):
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[figure] {path}")


def plot():
    pts = _load_sweep()
    if not pts:
        print("[skip] no sweep data yet")
        return
    pts.sort(key=lambda x: x[1])
    size_lbl = [p[0] for p in pts]
    params = np.array([p[1] for p in pts])
    layers = np.array([p[2] for p in pts])
    itl = np.array([p[3] for p in pts])
    ttft = np.array([p[4] for p in pts])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].loglog(params, itl, "o-", color="steelblue")
    axes[0].set_xlabel("Parameters (B)")
    axes[0].set_ylabel("Mean decode ITL (ms/token)")
    axes[0].set_title("Decode latency vs model size")
    for s, x, y in zip(size_lbl, params, itl):
        axes[0].annotate(s, (x, y), textcoords="offset points", xytext=(6, 6))

    axes[1].loglog(layers, ttft, "s-", color="darkorange")
    axes[1].set_xlabel("Transformer layers")
    axes[1].set_ylabel("TTFT (ms)")
    axes[1].set_title("Prefill cost (TTFT) vs layer count")
    for s, x, y in zip(size_lbl, layers, ttft):
        axes[1].annotate(s, (x, y), textcoords="offset points", xytext=(6, 6))

    for ax in axes:
        ax.grid(True, which="both", ls=":", alpha=0.4)
    _save(fig, "size_scaling.png")


if __name__ == "__main__":
    plot()
