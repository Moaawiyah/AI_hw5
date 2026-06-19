"""AirLLM runner (MLX backend on macOS) with quantization as a parameter (§5.3).

Uses model_generate()'s token generator directly to capture per-token timing,
so TTFT (Prefill/compute-bound) and ITL/TPOT (Decode/memory-bound) are separated.
"""
import time

import mlx.core as mx
import mlx.nn as nn

# Compatibility shim: airllm 2.11.0 calls mlx nn.Module.update(weights) with the
# default strict=True, but modern mlx rejects unknown keys (e.g. a spurious "bias"
# for bias-free Llama/Qwen attention). Default to strict=False so extra keys are
# skipped instead of raising. This is a known airllm<->mlx version drift issue.
_orig_update = nn.Module.update


def _lenient_update(self, parameters, strict=False):
    return _orig_update(self, parameters, strict=strict)


nn.Module.update = _lenient_update

from src import config
from src.hf_utils import local_dir
from src.metrics import peak_memory, now_ms, summarize_token_times
from src.token_count import exact_tokens_with_tokenizer


def _load(size_key: str, quant: str):
    from airllm import AutoModel
    path = str(local_dir(config.MODELS[size_key]["id"]))
    compression = config.QUANT_TO_COMPRESSION.get(quant)
    # Per-model shard dir so different models don't collide (§6.1).
    shards_path = config.SHARDS_DIR / size_key
    shards_path.mkdir(parents=True, exist_ok=True)
    # §6.1: use the general AutoModel entry point so airllm picks the right
    # architecture class (avoids Class mismatch). On macOS it routes to the
    # MLX-backed AirLLMLlamaMlx; on Linux/CUDA it maps Qwen/Llama/Mistral/etc.
    model = AutoModel.from_pretrained(
        path,
        dtype=mx.float16,
        compression=compression,
        hf_token=config.HF_TOKEN or None,
        layer_shards_saving_path=str(shards_path),
        max_seq_len=2048,
    )
    tok = model.get_tokenizer(hf_token=config.HF_TOKEN or None)
    return tok, model


def run(size_key: str, quant: str, prompt: str, max_new_tokens: int) -> dict:
    """Run AirLLM at a quant level; return a result dict (incl. failures)."""
    rec = {"scenario": "airllm", "size": size_key, "quant": quant, "prompt": prompt,
           "max_new_tokens": max_new_tokens, "ok": False}
    t0 = now_ms()
    try:
        tok, model = _load(size_key, quant)
        rec["load_ms"] = now_ms() - t0
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["wall_ms"] = now_ms() - t0
        return rec

    ids = tok(prompt, return_tensors=None)["input_ids"]
    x = mx.array([ids]) if not hasattr(ids, "shape") else mx.array(ids)
    if x.ndim == 1:
        x = x[None, :]

    token_times, tokens = [], []
    with peak_memory() as mt:
        gen = model.model_generate(x, temperature=0)
        for tok_id in gen:
            token_times.append(now_ms() - t0)
            tokens.append(int(tok_id.item()))
            if len(tokens) >= max_new_tokens:
                break
        rec["peak_rss_mb"] = mt.peak_rss_mb

    output = tok.decode(tokens) if tokens else ""
    rec["output"] = output
    rec["ok"] = True
    rec["wall_ms"] = now_ms() - t0
    rec["timing"] = summarize_token_times(token_times)
    rec["tokens"] = exact_tokens_with_tokenizer(tok, prompt, output)
    return rec
