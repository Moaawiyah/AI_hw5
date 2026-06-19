"""Ollama runner (GGUF format) — the assignment's other baseline path (§5.2).

Streams tokens from the local Ollama server REST API to capture TTFT (Prefill)
and ITL/TPOT (Decode), mirroring the other runners. Default Ollama tags are Q4_K_M
GGUF, so this also serves as a quantization comparison point (covers 'GGUF' keyword).
"""
import json
import time
import urllib.request

from src import config
from src.metrics import peak_memory, now_ms, summarize_token_times
from src.token_count import request_tokens

OLLAMA_URL = "http://localhost:11434/api/generate"

# Map our quant label -> ollama tag suffix for qwen2.5 (GGUF quant levels).
# The default `qwen2.5:14b` tag IS q4_K_M; other quants use `14b-instruct-<q>`.
QUANT_TO_TAG = {
    "q8": "instruct-q8_0",
    "q6": "instruct-q6_",
    "q5": "instruct-q5_0",
    "q4": "DEFAULT_Q4KM",   # the default qwen2.5:<size> tag is already Q4_K_M
    "q3": "instruct-q3_",
    "q2": "instruct-q2_K",
}


def _model_name(size_key: str, quant: str = "q4") -> str:
    """Map our size key + quant label to the Ollama qwen2.5 tag."""
    suffix = QUANT_TO_TAG.get(quant, "DEFAULT_Q4KM")
    base = size_key.lower()
    if suffix == "DEFAULT_Q4KM":
        return f"qwen2.5:{base}"          # default tag = Q4_K_M
    return f"qwen2.5:{base}-{suffix}"


def _stream(model: str, prompt: str, max_new_tokens: int, num_ctx: int = 2048):
    """Yield (token_text) from the Ollama generate stream.

    num_ctx caps the KV cache. The default 32k context balloons a 14B model to
    ~15GB (out of 16GB unified RAM) -> swap thrash. 2048 keeps it comfortably in RAM.
    """
    payload = json.dumps({
        "model": model, "prompt": prompt, "stream": True,
        "options": {"num_predict": max_new_tokens, "temperature": 0,
                    "num_ctx": num_ctx},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        for line in resp:
            if not line.strip():
                continue
            chunk = json.loads(line)
            if chunk.get("response"):
                yield chunk["response"]
            if chunk.get("done"):
                break


def run(size_key: str, prompt: str, max_new_tokens: int, tag: str = "q4",
        num_ctx: int = 2048) -> dict:
    """Run via Ollama; return a result dict (incl. failures)."""
    model = _model_name(size_key, tag)
    rec = {"scenario": "ollama", "size": size_key, "quant": tag, "model": model,
           "num_ctx": num_ctx, "prompt": prompt, "max_new_tokens": max_new_tokens,
           "ok": False}
    t0 = now_ms()
    try:
        # warm connectivity
        urllib.request.urlopen(f"http://localhost:11434/api/tags", timeout=10).close()
        rec["load_ms"] = 0.0
    except Exception as e:
        rec["error"] = f"ollama unreachable: {type(e).__name__}: {e}"
        rec["wall_ms"] = now_ms() - t0
        return rec

    token_times, output = [], ""
    with peak_memory() as mt:
        try:
            for piece in _stream(model, prompt, max_new_tokens, num_ctx):
                token_times.append(now_ms() - t0)
                output += piece
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["wall_ms"] = now_ms() - t0
            return rec
        rec["peak_rss_mb"] = mt.peak_rss_mb

    rec["output"] = output
    rec["ok"] = True
    rec["wall_ms"] = now_ms() - t0
    rec["timing"] = summarize_token_times(token_times)
    rec["tokens"] = request_tokens(prompt, output)
    return rec
