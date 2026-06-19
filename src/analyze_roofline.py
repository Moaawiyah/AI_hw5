"""Roofline model (§3 advanced aspiration): plot arithmetic intensity vs peak
FLOPs/memory-bandwidth to visualize compute-bound vs memory-bound crossover."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config

# Apple M3 approximate specs (state assumptions in report).
M3_PEAK_TFLOPS_FP16 = 4.6      # ~4.6 TFLOPS FP16 (GPU)
M3_MEM_BANDWIDTH_GB_S = 100.0  # ~100 GB/s unified memory bandwidth


def _save(fig, name):
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[figure] {path}")


def plot(points=None):
    """points: list of (label, arithmetic_intensity_flops_per_byte)."""
    intensity = np.logspace(-1, 3, 500)  # FLOP/byte
    peak_flops = M3_PEAK_TFLOPS_FP16 * 1e12
    bw_bytes = M3_MEM_BANDWIDTH_GB_S * 1e9
    ridge = peak_flops / bw_bytes  # FLOP/byte at the knee

    achievable = np.minimum(peak_flops, intensity * bw_bytes) / 1e12  # TFLOPS

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.loglog(intensity, achievable, color="steelblue", lw=2, label="M3 roofline")
    ax.axvline(ridge, color="red", ls="--", label=f"ridgepoint ≈ {ridge:.1f} FLOP/byte")
    ax.set_xlabel("Arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("Achievable throughput (TFLOP/s)")
    ax.set_title("M3 Roofline: compute-bound (right) vs memory-bound (left)")
    ax.grid(True, which="both", ls=":", alpha=0.4)

    if points:
        for label, ai in points:
            perf = min(peak_flops, ai * bw_bytes) / 1e12
            ax.plot(ai, perf, "o")
            ax.annotate(label, (ai, perf), textcoords="offset points",
                        xytext=(6, 6), fontsize=9)
    ax.legend()
    _save(fig, "roofline.png")


# Representative points: decode is memory-bound (low AI), prefill more compute-bound.
DEFAULT_POINTS = [
    ("Prefill (FP16)", 50),
    ("Decode (FP16)", 2),
    ("Decode (Q4)", 1),
]


def main():
    plot(DEFAULT_POINTS)


if __name__ == "__main__":
    main()
