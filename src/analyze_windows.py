"""Generate the Windows benchmark summary and figures from results/windows/."""
from pathlib import Path
import os
import statistics as stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config, report

RESULTS = Path(os.environ.get("WINDOWS_RESULTS_DIR", config.RESULTS_DIR / "windows"))
SIZES = {"0.5B": 0.5, "1.5B": 1.5, "14B": 14.0}


def load():
    return report.load_results_md(RESULTS)


def _successful(records, scenario, **filters):
    return [r for r in records if r.get("ok") and r.get("scenario") == scenario
            and all(str(r.get(k)).lower() == str(v).lower() for k, v in filters.items())]


def aggregate(records):
    air = {size: _successful(records, "airllm", size=size)[-1] for size in SIZES}
    quants = {}
    for quant in ("q2", "q4", "q8"):
        runs = sorted(_successful(records, "ollama", quant=quant),
                      key=lambda r: r.get("repeat", 0))
        cold, warm = runs[0], runs[1:]
        quants[quant] = {
            "cold_ttft": cold["timing"]["ttft_ms"],
            "cold_wall": cold["wall_ms"],
            "warm_ttft": stats.mean(r["timing"]["ttft_ms"] for r in warm),
            "warm_itl": stats.mean(r["timing"]["itl_mean_ms"] for r in warm),
            "warm_tps": stats.mean(r["timing"]["throughput_tps"] for r in warm),
            "warm_wall": stats.mean(r["wall_ms"] for r in warm),
        }
    baseline = [r for r in records if r.get("scenario") == "baseline"][-1]
    return baseline, air, quants


def _save(fig, name):
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[figure] {path}")


def plot_quant(quants):
    labels = ["Q2_K", "Q4_K_M", "Q8_0"]
    rows = [quants[q] for q in ("q2", "q4", "q8")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, key, title, color in zip(
            axes, ("cold_ttft", "warm_itl", "warm_tps"),
            ("Cold TTFT (ms)", "Warm ITL (ms/token)", "Warm throughput (tok/s)"),
            ("#9467bd", "#8c564b", "#17becf")):
        ax.bar(labels, [r[key] for r in rows], color=color)
        ax.set_title(title)
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("Windows RTX 4060 — Qwen2.5-14B Ollama quantization")
    _save(fig, "windows_quant_sweep.png")


def plot_paths(air, quants):
    labels = ["AirLLM\nFP16", "Ollama\nQ2", "Ollama\nQ4", "Ollama\nQ8"]
    tps = [air["14B"]["timing"]["throughput_tps"]] + [
        quants[q]["warm_tps"] for q in ("q2", "q4", "q8")]
    itl = [air["14B"]["timing"]["itl_mean_ms"]] + [
        quants[q]["warm_itl"] for q in ("q2", "q4", "q8")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(labels, tps, color=["#d62728", "#2ca02c", "#1f77b4", "#9467bd"])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Tokens/s (log scale)")
    axes[0].set_title("Decode throughput")
    axes[1].bar(labels, itl, color=["#d62728", "#2ca02c", "#1f77b4", "#9467bd"])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("ITL ms/token (log scale)")
    axes[1].set_title("Decode latency")
    _save(fig, "windows_path_comparison.png")


def plot_scaling(air):
    labels = list(SIZES)
    params = [SIZES[s] for s in labels]
    itl = [air[s]["timing"]["itl_mean_ms"] for s in labels]
    rss = [air[s]["peak_rss_mb"] for s in labels]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].loglog(params, itl, "o-", color="steelblue")
    axes[0].set(xlabel="Parameters (B)", ylabel="ITL (ms/token)",
                title="AirLLM decode scaling")
    axes[1].plot(params, rss, "s-", color="darkorange")
    axes[1].set(xlabel="Parameters (B)", ylabel="Peak Python RSS (MB)",
                title="AirLLM memory scaling")
    for ax in axes:
        ax.grid(True, which="both", ls=":", alpha=.4)
    _save(fig, "windows_size_scaling.png")


def write_summary(baseline, air, quants):
    lines = ["# Windows benchmark — Qwen2.5", "",
             "Hardware: ASUS TUF Gaming F15, i7-13700H, 23.6 GB RAM, "
             "RTX 4060 Laptop GPU (8 GB).", "",
             "| Path | TTFT | ITL | Throughput | Wall | Outcome |",
             "|---|---:|---:|---:|---:|---|"]
    lines.append(f"| Baseline 14B FP16 | — | — | — | {baseline['wall_ms']/1000:.2f} s | CUDA OOM |")
    a = air["14B"]
    lines.append(f"| AirLLM 14B FP16 | {a['timing']['ttft_ms']/1000:.2f} s | "
                 f"{a['timing']['itl_mean_ms']/1000:.2f} s | "
                 f"{a['timing']['throughput_tps']:.2f} tok/s | "
                 f"{a['wall_ms']/60000:.2f} min | ran |")
    for q, name in (("q2", "Q2_K"), ("q4", "Q4_K_M"), ("q8", "Q8_0")):
        x = quants[q]
        lines.append(f"| Ollama 14B {name} (warm mean) | {x['warm_ttft']/1000:.2f} s | "
                     f"{x['warm_itl']:.2f} ms | {x['warm_tps']:.2f} tok/s | "
                     f"{x['warm_wall']/1000:.2f} s | ran |")
    lines += ["", "## AirLLM size scaling", "",
              "| Size | Peak RSS | TTFT | ITL | Throughput | Wall |",
              "|---|---:|---:|---:|---:|---:|"]
    for size in SIZES:
        r, t = air[size], air[size]["timing"]
        lines.append(f"| {size} | {r['peak_rss_mb']:.0f} MB | {t['ttft_ms']/1000:.2f} s | "
                     f"{t['itl_mean_ms']/1000:.2f} s | {t['throughput_tps']:.2f} tok/s | "
                     f"{r['wall_ms']/60000:.2f} min |")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    baseline, air, quants = aggregate(load())
    write_summary(baseline, air, quants)
    plot_quant(quants)
    plot_paths(air, quants)
    plot_scaling(air)


if __name__ == "__main__":
    main()
