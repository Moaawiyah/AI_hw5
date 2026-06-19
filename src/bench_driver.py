"""CLI orchestrator: run scenarios, repeat, persist raw JSON (§5.4, §6.1 'keep raw numbers')."""
import json
import time
import argparse

from src import config
from experiments import prompts as exp_prompts
from src import run_baseline, run_airllm, run_ollama, run_sweep


def _save(rec: dict, tag: str) -> None:
    fname = f"{tag}__{int(time.time())}.json"
    path = config.RESULTS_DIR / fname
    with open(path, "w") as f:
        json.dump(rec, f, indent=2, default=str)
    print(f"[saved] {path}")


def cmd_baseline(args):
    sc = exp_prompts.SCENARIOS["short"]
    for r in range(args.repeats):
        rec = run_baseline.run(args.size, sc["prompt"], sc["max_new_tokens"])
        rec["repeat"] = r
        _save(rec, f"baseline_{args.size}")


def cmd_airllm(args):
    sc = exp_prompts.SCENARIOS["short"]
    for quant in (args.quants or config.QUANT_LEVELS):
        for r in range(args.repeats):
            rec = run_airllm.run(args.size, quant, sc["prompt"], sc["max_new_tokens"])
            rec["repeat"] = r
            _save(rec, f"airllm_{args.size}_{quant}")


def cmd_sweep(args):
    sc = exp_prompts.SCENARIOS["short"]
    for size in (config.SWEEP_SIZES if args.all else [args.size]):
        rec = run_airllm.run(size, "fp16", sc["prompt"], args.max_new_tokens)
        rec["scenario"] = "size_sweep"
        _save(rec, f"sweep_{size}")


def cmd_ollama(args):
    sc = exp_prompts.SCENARIOS["short"]
    for r in range(args.repeats):
        rec = run_ollama.run(args.size, sc["prompt"], sc["max_new_tokens"])
        rec["repeat"] = r
        _save(rec, f"ollama_{args.size}")


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

    o = sub.add_parser("ollama", help="Ollama (GGUF) run")
    o.add_argument("--size", default=config.SUBJECT)
    o.add_argument("--repeats", type=int, default=config.REPEATS)
    o.set_defaults(func=cmd_ollama)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
