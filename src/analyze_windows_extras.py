"""Illustrative RTX 4060 roofline and Windows cost-scenario figures."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.prompts import PRIMARY_PROMPT
from src import config, economics

RTX4060_FP16_TFLOPS = 15.0
RTX4060_BANDWIDTH_GB_S = 256.0
ASSUMED_CAPEX_USD = 1500.0


def _save(fig, name):
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[figure] {path}")


def roofline():
    intensity = np.logspace(-1, 3, 500)
    peak = RTX4060_FP16_TFLOPS * 1e12
    bandwidth = RTX4060_BANDWIDTH_GB_S * 1e9
    ridge = peak / bandwidth
    achievable = np.minimum(peak, intensity * bandwidth) / 1e12
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.loglog(intensity, achievable, lw=2, label="RTX 4060 Laptop roofline")
    ax.axvline(ridge, color="red", ls="--", label=f"ridge ≈ {ridge:.1f} FLOP/byte")
    for label, ai in (("Prefill FP16", 50), ("Decode FP16", 2), ("Decode Q2", 1)):
        perf = min(peak, ai * bandwidth) / 1e12
        ax.plot(ai, perf, "o")
        ax.annotate(label, (ai, perf), xytext=(6, 6), textcoords="offset points")
    ax.set(xlabel="Arithmetic intensity (FLOP/byte)", ylabel="TFLOP/s",
           title="Illustrative RTX 4060 Laptop roofline")
    ax.grid(True, which="both", ls=":", alpha=.4)
    ax.legend()
    _save(fig, "windows_roofline.png")


def _cost(volumes, seconds, watts):
    old_capex, old_watts = config.HW_CAPEX_USD, config.HW_LOAD_W
    config.HW_CAPEX_USD, config.HW_LOAD_W = ASSUMED_CAPEX_USD, watts
    output = ("Paging moves unused memory pages to disk and back as needed, "
              "letting programs use more memory than physically present.")
    curves = economics.cumulative(volumes, PRIMARY_PROMPT, output, seconds)
    config.HW_CAPEX_USD, config.HW_LOAD_W = old_capex, old_watts
    return curves


def breakeven():
    volumes = np.logspace(1, 7, 300)
    q2 = _cost(volumes, 6.64, 120)
    q4 = _cost(volumes, 10.52, 120)
    air = _cost(volumes, 2667.43, 45)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.loglog(volumes, q2["api"], lw=2, label="API (GPT-4o assumption)")
    ax.loglog(volumes, q2["onprem"], lw=2, label="Windows Ollama Q2")
    ax.loglog(volumes, q4["onprem"], lw=2, label="Windows Ollama Q4")
    ax.loglog(volumes, air["onprem"], lw=2, label="Windows AirLLM 14B")
    for curves, label in ((q2, "Q2"), (q4, "Q4")):
        be = economics.find_breakeven(curves["api"], curves["onprem"], volumes)
        if be:
            ax.axvline(be, color="grey", ls=":", alpha=.5)
            ax.text(be, 70, f"{label}: {be/1000:.0f}k/mo", rotation=90, va="top")
    ax.set(xlabel="Requests/month", ylabel="Monthly cost (USD)",
           title="Windows cost scenario — assumed $1,500 CAPEX")
    ax.grid(True, which="both", ls=":", alpha=.35)
    ax.legend()
    _save(fig, "windows_breakeven.png")


def main():
    roofline()
    breakeven()


if __name__ == "__main__":
    main()
