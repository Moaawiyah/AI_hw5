"""Break-even figure (§5.5): cumulative cost vs monthly volume for API / On-Prem /
Cloud GPU, with the crossover annotated."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config, economics
from experiments.prompts import PRIMARY_PROMPT


def _save(fig, name):
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[figure] {path}")


def plot(onprem_seconds=15.0, sample_output=None):
    sample_output = sample_output or (
        "Paging moves unused memory pages to disk and back as needed, "
        "letting programs use more memory than physically present."
    )
    vols = np.logspace(1, 6, 60)
    c = economics.cumulative(vols, PRIMARY_PROMPT, sample_output, onprem_seconds)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.loglog(vols, c["api"], label="API (gpt-4o)", lw=2)
    ax.loglog(vols, c["api_cached"], "--", label="API w/ prompt-cache (80%)", lw=2)
    ax.loglog(vols, c["onprem"], label="On-Prem (M3)", lw=2)
    ax.loglog(vols, c["cloud_gpu"], label="Cloud GPU", lw=2)

    be = economics.find_breakeven(c["api"], c["onprem"], vols)
    if be:
        ax.axvline(be, color="grey", ls=":")
        ax.annotate(f"break-even\n≈{be:.0f}/mo", (be, c["onprem"][np.searchsorted(vols, be)]),
                    textcoords="offset points", xytext=(8, -30), fontsize=9)

    ax.set_xlabel("Requests / month")
    ax.set_ylabel("Cumulative monthly cost (USD)")
    ax.set_title("On-Prem vs API vs Cloud GPU — break-even analysis")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend()
    _save(fig, "breakeven.png")


if __name__ == "__main__":
    plot()
