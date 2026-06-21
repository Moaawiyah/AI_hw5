"""Markdown writers/parsers for result files (§7 deliverable).

Result files in results/ are pure Markdown tables (one per run) — they render
nicely on GitHub and ``parse_record_md`` reads them back so analyze_*.py can
aggregate. The aggregated table is also Markdown.
"""
import math
import re
from pathlib import Path

_SECTIONS = {"Run": None, "Timing": "timing", "Tokens": "tokens", "Quality": "quality"}
_ROW_RE = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|$")


def _fmt(v, nd=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _coerce(s):
    """Infer bool/int/float/str from a string read back from a table cell."""
    s = s.strip()
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    if s in ("—", "", "None"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _table(title, rows):
    out = [f"## {title}", "", "| Field | Value |", "|---|---:|"]
    for k, v in rows:
        out.append(f"| {k} | {_fmt(v)} |")
    return out + [""]


def format_record_md(rec: dict) -> str:
    """Render one benchmark record as a Markdown document (pure tables)."""
    sc = rec.get("scenario", "?")
    model = rec.get("model") or rec.get("size", "?")
    quant = rec.get("quant", "")
    title = f"{sc.capitalize()} — {model}" + (f" ({quant})" if quant else "")
    t = rec.get("timing") or {}
    tok = rec.get("tokens") or {}
    qual = rec.get("quality") or {}

    L = [f"# {title}", "", f"**Status:** {'success' if rec.get('ok') else 'failed'}", ""]
    if rec.get("error"):
        L += [f"**Error:** `{rec['error']}`", ""]
    L += _table("Run", [("scenario", sc), ("size", rec.get("size")),
                         ("quant", quant or None), ("model", rec.get("model", "")),
                         ("ok", rec.get("ok")), ("wall_ms", rec.get("wall_ms")),
                         ("load_ms", rec.get("load_ms")),
                         ("peak_rss_mb", rec.get("peak_rss_mb")),
                         ("estimated_kwh", rec.get("estimated_kwh")),
                         ("repeat", rec.get("repeat"))])
    L += _table("Timing", [("ttft_ms", t.get("ttft_ms")),
                            ("itl_mean_ms", t.get("itl_mean_ms")),
                            ("itl_p50_ms", t.get("itl_p50_ms")),
                            ("throughput_tps", t.get("throughput_tps")),
                            ("n_tokens", t.get("n_tokens"))])
    L += _table("Tokens", [("input_tokens", tok.get("input_tokens")),
                            ("output_tokens", tok.get("output_tokens")),
                            ("total_tokens", tok.get("total_tokens"))])
    if qual:
        L += _table("Quality", list(qual.items()))
    out = (rec.get("output") or "").strip()
    if out:
        L += ["## Output", "", "> " + out.replace("\n", "\n> "), ""]
    return "\n".join(L) + "\n"


def parse_record_md(text: str) -> dict:
    """Parse a result Markdown file back into a (nested) dict."""
    rec, section = {}, None
    for line in text.splitlines():
        h = re.match(r"^##\s+(.+?)\s*$", line)
        if h:
            section = _SECTIONS.get(h.group(1).strip(), h.group(1).strip().lower())
            continue
        m = _ROW_RE.match(line)
        if not m:
            continue
        key, val = m.group(1).strip(), m.group(2).strip()
        if key.lower() == "field" or set(key) <= {"-", ":"}:
            continue
        bucket = rec if section is None else rec.setdefault(section, {})
        bucket[key] = _coerce(val)
    return rec


def format_summary_md(rows: list) -> str:
    """Render rows as a Markdown summary table. Accepts flat (pandas agg) or
    parsed per-run records (with timing nested)."""
    cols = ["scenario", "size", "quant", "ok", "ttft_ms", "itl_mean_ms",
            "throughput_tps", "peak_rss_mb", "wall_ms", "estimated_kwh"]
    L = ["# Aggregated results", "",
         "| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        flat = dict(r)
        for k, v in (r.get("timing") or {}).items():
            flat.setdefault(k, v)
        L.append("| " + " | ".join(_fmt(flat.get(c)) for c in cols) + " |")
    return "\n".join(L) + "\n"


def write_record(rec: dict, path) -> None:
    Path(path).write_text(format_record_md(rec))


def write_summary(rows: list, path) -> None:
    Path(path).write_text(format_summary_md(rows))


def load_results_md(results_dir) -> list:
    """Load every *.md result file (excluding summary.md/README.md) into dicts."""
    out = []
    for fp in sorted(Path(results_dir).glob("*.md")):
        if fp.name in ("summary.md", "README.md"):
            continue
        try:
            out.append(parse_record_md(fp.read_text()))
        except Exception as e:
            print(f"[warn] skip {fp.name}: {e}")
    return out
