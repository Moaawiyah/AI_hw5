"""Aggregate raw result Markdown files and plot comparative charts (§5.4).

Figures:
  - path_comparison.png: baseline(OOM) vs AirLLM vs Ollama-q4 headline comparison.
  - quant_sweep.png:      GGUF quantization sweep (q2/q4/q8) TTFT/ITL/throughput.
  - summary.md:           aggregated numeric table (Markdown).
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config, report


def load_results() -> pd.DataFrame:
    rows = []
    for r in report.load_results_md(config.RESULTS_DIR):
        t = r.get("timing", {}) or {}
        rows.append({
            "scenario": r.get("scenario"),
            "size": str(r.get("size")),
            "quant": r.get("quant", "fp16"),
            "model": r.get("model", ""),
            "ok": r.get("ok"),
            "ttft_ms": t.get("ttft_ms"),
            "itl_mean_ms": t.get("itl_mean_ms"),
            "throughput_tps": t.get("throughput_tps"),
            "peak_rss_mb": r.get("peak_rss_mb"),
            "wall_ms": r.get("wall_ms"),
            "estimated_kwh": r.get("estimated_kwh"),
            "error": (r.get("error") or "")[:40],
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        # Failed runs use an em dash in Markdown for unavailable measurements.
        # Normalize placeholders to NaN so aggregation works across pandas versions.
        numeric = [
            "ttft_ms", "itl_mean_ms", "throughput_tps", "peak_rss_mb",
            "wall_ms", "estimated_kwh",
        ]
        df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    return df


def _save(fig, name):
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[figure] {path}")


def plot_path_comparison(df: pd.DataFrame):
    """Headline: peak-RAM and throughput across the 3 running approaches."""
    rows = []
    if config.SUBJECT:
        # AirLLM subject
        a = df[(df.scenario == "airllm") & (df["size"] == config.SUBJECT)]
        if len(a):
            r = a.iloc[0]
            rows.append((f"AirLLM\n({config.SUBJECT} FP16)", r.peak_rss_mb, r.throughput_tps))
    o = df[df.scenario == "ollama"]
    if len(o):
        r = o[o.quant == "q4"]
        if len(r):
            r = r.iloc[0]
            subj = config.SUBJECT
            rows.append((f"Ollama GGUF\n(Qwen2.5 {subj} Q4)", r.peak_rss_mb, r.throughput_tps))
    rows.append((f"Baseline\n({config.SUBJECT} FP16)", config.BASELINE_OOM_MB, 0.0))

    labels = [x[0] for x in rows]
    ram = [x[1] for x in rows]
    tput = [x[2] or 0 for x in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.bar(labels, ram, color=["#d62728", "#2ca02c", "#1f77b4"])
    ax1.set_ylabel("Peak RAM (MB)")
    ax1.set_title("Memory footprint (lower = better)")
    ax1.axhline(config.HW_RAM_MB, color="grey", ls="--", lw=1)
    ax1.text(0.1, config.HW_RAM_MB, " 16GB unified RAM", color="grey", fontsize=8)
    ax2.bar(labels, tput, color=["#d62728", "#2ca02c", "#1f77b4"])
    ax2.set_ylabel("Throughput (tok/s)")
    ax2.set_title("Decode throughput (higher = better)")
    _save(fig, "path_comparison.png")


def plot_quant_sweep(df: pd.DataFrame):
    o = df[df.scenario == "ollama"].copy()
    o = o[o["size"].str.lower() == "14b"]
    o["order"] = o.quant.map({"q2": 0, "q4": 1, "q8": 2})
    o = o.sort_values("order").dropna(subset=["order"])
    if o.empty:
        print("[skip] no ollama quant data")
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, col, title, colr in zip(
        axes, ["ttft_ms", "itl_mean_ms", "throughput_tps"],
        ["TTFT (ms)", "Mean decode ITL (ms)", "Throughput (tok/s)"],
        ["#9467bd", "#8c564b", "#17becf"]):
        ok = o[o.ok]
        ax.bar(ok.quant, ok[col], color=colr)
        # mark failures
        for _, fr in o[~o.ok].iterrows():
            ax.axvline(fr.order - 0.4 + 0.4, color="red", alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel("GGUF quant level")
    title = f"Quantization sweep — Qwen2.5-{config.SUBJECT} via Ollama (q8 = OOM/swap)"
    fig.suptitle(title)
    _save(fig, "quant_sweep.png")


def write_table(df: pd.DataFrame):
    agg = df.groupby(["scenario", "size", "quant"]).agg(
        ok=("ok", "first"), ttft_ms=("ttft_ms", "mean"),
        itl_mean_ms=("itl_mean_ms", "mean"), throughput_tps=("throughput_tps", "mean"),
        peak_rss_mb=("peak_rss_mb", "mean"), wall_ms=("wall_ms", "mean"),
        estimated_kwh=("estimated_kwh", "mean")).round(5).reset_index()
    rows = agg.to_dict(orient="records")
    report.write_summary(rows, config.RESULTS_DIR / "summary.md")
    print(f"[table] {config.RESULTS_DIR / 'summary.md'}")
    print(agg.to_string(index=False))


def main():
    df = load_results()
    write_table(df)
    plot_path_comparison(df)
    plot_quant_sweep(df)


if __name__ == "__main__":
    main()
