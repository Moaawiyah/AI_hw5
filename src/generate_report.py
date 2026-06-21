"""Generate reports/REPORT.md — the data-driven technical report deliverable (§7, §8).

Reads the result Markdown files in results/, computes the headline comparison,
quantization sweep, scaling, and economics tables, and writes a single
aggregated Markdown report. Run after benchmarks: `python -m src.generate_report`.
"""
import numpy as np

from src import config, report, economics
from experiments.prompts import PRIMARY_PROMPT


def _fmt(v, nd=2):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _headline(recs):
    """3-row table: baseline / AirLLM / Ollama (peak RAM, throughput, outcome)."""
    def pick(scenario, **filt):
        for r in recs:
            if r.get("scenario") != scenario:
                continue
            if all(str(r.get(k, "")).lower() == str(v).lower() for k, v in filt.items()):
                return r
        return None
    base = pick("baseline")
    air = pick("airllm", quant="fp16")
    ol = pick("ollama", quant="q4")
    rows = []
    if base:
        rows.append(("Baseline (FP16)", base.get("peak_rss_mb"), None,
                     "OOM / swap thrash" if not base.get("ok") else "ran"))
    if air:
        t = air.get("timing") or {}
        rows.append(("AirLLM (FP16)", air.get("peak_rss_mb"),
                     t.get("throughput_tps"), "ran, memory-bound"))
    if ol:
        t = ol.get("timing") or {}
        rows.append(("Ollama GGUF (Q4)", ol.get("peak_rss_mb"),
                     t.get("throughput_tps"), "ran, comfortable"))
    body = ["| Path | Peak RAM (MB) | Throughput (tok/s) | Outcome |",
            "|---|---:|---:|---|"]
    for name, ram, tput, out in rows:
        body.append(f"| {name} | {_fmt(ram, 0)} | {_fmt(tput, 2)} | {out} |")
    return "\n".join(body)


def _economics():
    sample = ("Paging moves unused memory pages to disk and back as needed, "
              "letting programs use more memory than physically present.")
    vols = np.logspace(1, 6, 60)
    lines = ["| Path | Latency/req | Break-even vs GPT-4o |",
             "|---|---:|---:|"]
    for secs, label in [(15, "Ollama Q4 (~15s)"), (600, "AirLLM (~600s)")]:
        c = economics.cumulative(vols, PRIMARY_PROMPT, sample, onprem_seconds=secs)
        be = economics.find_breakeven(c["api"], c["onprem"], vols)
        lines.append(f"| {label} | {secs}s | {_fmt(be, 0)} req/mo |"
                     if be else f"| {label} | {secs}s | never |")
    return "\n".join(lines)


def generate() -> str:
    recs = report.load_results_md(config.RESULTS_DIR)
    results_rel = config.RESULTS_DIR.relative_to(config.ROOT)
    # Strip the leading H1 from format_summary_md so it nests cleanly under §2.
    summary_body = "\n".join(report.format_summary_md(recs).splitlines()[2:])
    L = ["# EX05 — Technical Report (data-driven)", "",
         f"Subject model: **{config.MODELS[config.SUBJECT]['id']}**  ",
         "Hardware: Apple M3 MacBook Pro, 16 GB unified memory.  ",
         f"Generated from `{results_rel}/` Markdown result files.", "",
         "## 1. Headline comparison (§5.4)", "", _headline(recs), "",
         "## 2. Aggregated results", "", summary_body, "",
         "## 3. Economics — On-Prem vs API (§5.5)", "", _economics(), "",
         "## 4. Figures", "",
         "- ![Path comparison](../figures/path_comparison.png)",
         "- ![Quant sweep](../figures/quant_sweep.png)",
         "- ![Size scaling](../figures/size_scaling.png)",
         "- ![Roofline](../figures/roofline.png)",
         "- ![Break-even](../figures/breakeven.png)", "",
         "## 5. Reproduction", "",
         "See `../README.md` §7 for full instructions. "
         f"Raw per-run data is in `{results_rel}/`.",
         ""]
    return "\n".join(L)


if __name__ == "__main__":
    out = config.ROOT / "reports" / "REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(generate())
    print(f"[report] {out}")
