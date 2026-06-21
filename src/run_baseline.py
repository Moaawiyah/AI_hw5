"""Baseline runner: load model directly via transformers on MPS (§5.2).

Expected on a 16GB M3 with a 14B FP16 model: OOM or severe swap thrash.
We capture that outcome as the documented baseline rather than crashing.
"""
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from src import config
from src.hf_utils import local_dir
from src.metrics import peak_memory, now_ms, summarize_token_times
from src.token_count import exact_tokens_with_tokenizer


def _load(size_key: str):
    path = str(local_dir(config.MODELS[size_key]["id"]))
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16,
        device_map="mps",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tok, model


def run(size_key: str, prompt: str, max_new_tokens: int) -> dict:
    """Run baseline; return a result dict (including failures)."""
    rec = {"scenario": "baseline", "size": size_key, "prompt": prompt,
           "max_new_tokens": max_new_tokens, "ok": False}
    t0 = now_ms()
    try:
        tok, model = _load(size_key)
        rec["load_ms"] = now_ms() - t0
    except Exception as e:
        rec["ok"] = False
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["wall_ms"] = now_ms() - t0
        rec["bottleneck"] = "load_failed"
        return rec

    inputs = tok(prompt, return_tensors="pt").to("mps")
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                      streamer=streamer, pad_token_id=tok.eos_token_id)

    token_times, output = [], ""
    with peak_memory() as mt:
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()
        for piece in streamer:
            token_times.append(now_ms() - t0)
            output += piece
        thread.join()
        rec["peak_rss_mb"] = mt.peak_rss_mb
    rec["output"] = output
    rec["ok"] = True
    rec["wall_ms"] = now_ms() - t0
    rec["timing"] = summarize_token_times(token_times)
    rec["tokens"] = exact_tokens_with_tokenizer(tok, prompt, output)
    return rec
