"""CLI orchestrator: run scenarios, repeat, persist results as Markdown (§5.4, §6.1)."""
import time
import argparse

from src import config, report
from src.metrics import estimate_power_kwh
from experiments import prompts as exp_prompts
from src import run_baseline, run_airllm, run_ollama


def _annotate(rec: dict) -> dict:
    """Add power estimate and quality score to a completed result record."""
    wall = rec.get("wall_ms") or 0.0
    rec["estimated_kwh"] = estimate_power_kwh(wall, config.HW_LOAD_W)
    rec["quality"] = exp_prompts.score_output(rec.get("output") or "")
    return rec


def _save(rec: dict, tag: str) -> None:
    fname = f"{tag}__{int(time.time())}.md"
    path = config.RESULTS_DIR / fname
    report.write_record(rec, path)
    print(f"[saved] {path}")


def cmd_baseline(args):
    sc = exp_prompts.SCENARIOS["short"]
    for r in range(args.repeats):
        rec = run_baseline.run(args.size, sc["prompt"], sc["max_new_tokens"])
        rec["repeat"] = r
        _save(_annotate(rec), f"baseline_{args.size}")


def cmd_airllm(args):
    sc = exp_prompts.SCENARIOS["short"]
    for quant in (args.quants or config.QUANT_LEVELS):
        for r in range(args.repeats):
            rec = run_airllm.run(args.size, quant, sc["prompt"], sc["max_new_tokens"])
            rec["repeat"] = r
            _save(_annotate(rec), f"airllm_{args.size}_{quant}")


def cmd_sweep(args):
    sc = exp_prompts.SCENARIOS["short"]
    for size in (config.SWEEP_SIZES if args.all else [args.size]):
        rec = run_airllm.run(size, "fp16", sc["prompt"], args.max_new_tokens)
        rec["scenario"] = "size_sweep"
        _save(_annotate(rec), f"sweep_{size}")


def cmd_ollama(args):
    sc = exp_prompts.SCENARIOS["short"]
    for tag in args.quants:
        for r in range(args.repeats):
            rec = run_ollama.run(args.size, sc["prompt"], sc["max_new_tokens"], tag=tag)
            rec["repeat"] = r
            _save(_annotate(rec), f"ollama_{args.size}_{tag}")


def main():
    p = argparse.ArgumentParser(description="Benchmark orchestrator")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("baseline", help="direct transformers+MPS run")
    b.add_argument("--size", default=config.SUBJECT)
    b.add_argument("--repeats", type=int, default=config.REPEATS)
    b.set_defaults(func=cmd_baseline)

    a = sub.add_parser("airllm", help="AirLLM quant sweep")
    a.add_argument("--size", default=config.SUBJECT)
    a.add_argument("--quants", nargs="*", default=None)
    a.add_argument("--repeats", type=int, default=config.REPEATS)
    a.set_defaults(func=cmd_airllm)

    s = sub.add_parser("sweep", help="model-size scaling sweep")
    s.add_argument("--size", default=config.SUBJECT)
    s.add_argument("--all", action="store_true", help="sweep whole family")
    s.add_argument("--max-new-tokens", type=int, default=24)
    s.set_defaults(func=cmd_sweep)

    o = sub.add_parser("ollama", help="Ollama (GGUF) quant sweep")
    o.add_argument("--size", default="14b")
    o.add_argument("--quants", nargs="*", default=["q4"],
                   help="GGUF quant tags to sweep, e.g. --quants q2 q4 q8")
    o.add_argument("--repeats", type=int, default=config.REPEATS)
    o.set_defaults(func=cmd_ollama)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
