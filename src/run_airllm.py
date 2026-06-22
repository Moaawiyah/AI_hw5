"""AirLLM runner (PyTorch/CPU path) with Qwen2.5 support (§5.3).

AirLLM 2.11.0 on macOS hardcodes AutoModel -> AirLLMLlamaMlx, which only supports
Llama-style MHA. We bypass this by importing AirLLMQWen2 directly and running on
CPU. bitsandbytes (CUDA-only) is unavailable, so only fp16 is supported here.
"""
import threading

import torch
from transformers import TextIteratorStreamer

from src import config
from src.hf_utils import local_dir, untie_embeddings
from src.metrics import peak_memory, now_ms, summarize_token_times
from src.token_count import exact_tokens_with_tokenizer


def _stub_better_transformer():
    """optimum.bettertransformer was removed in optimum>=1.19.  airllm_base
    imports it at module level even though it's never called. Inject a stub."""
    import sys
    import types
    if "optimum.bettertransformer" not in sys.modules:
        mod = types.ModuleType("optimum.bettertransformer")
        class _BT:
            @staticmethod
            def transform(model):
                return model
        mod.BetterTransformer = _BT
        sys.modules["optimum.bettertransformer"] = mod


def _patch_mlx_persister():
    """MlxModelPersister.load_model renames PyTorch keys to MLX names, but
    AirLLMQWen2 (PyTorch path) needs the original names. Shards on disk already
    store PyTorch names — load them verbatim."""
    import numpy as np
    import torch
    from pathlib import Path
    from airllm.persist.mlx_model_persister import MlxModelPersister

    def _load_model(self, layer_name, path):
        data = np.load(str(Path(path) / (layer_name + ".mlx.npz")))
        return {k: torch.from_numpy(np.array(v)).to(torch.float16) for k, v in data.items()}

    MlxModelPersister.load_model = _load_model


def _patch_dynamic_cache():
    """transformers>=4.38 passes DynamicCache objects as past_key_values but
    AirLLM 2.11.0 expects tuples.  Patch get_past_key_values_cache_seq_len."""
    from airllm.airllm_base import AirLLMBaseModel

    def _seq_len(self, past_key_values):
        if hasattr(past_key_values, "get_seq_length"):   # DynamicCache
            return past_key_values.get_seq_length()
        return past_key_values[0][0].shape[2]            # legacy tuple

    AirLLMBaseModel.get_past_key_values_cache_seq_len = _seq_len


def _ensure_shard_index(model_dir):
    """airllm needs model.safetensors.index.json; single-shard Qwen2.5 (0.5B)
    only ships model.safetensors. Pre-write a minimal index."""
    import json
    from pathlib import Path
    from safetensors import safe_open
    p = Path(model_dir)
    idx, single = p / "model.safetensors.index.json", p / "model.safetensors"
    if idx.exists() or not single.exists():
        return
    with safe_open(str(single), framework="pt") as f:
        weight_map = {k: "model.safetensors" for k in f.keys()}
    idx.write_text(json.dumps({"metadata": {"total_size": single.stat().st_size},
                               "weight_map": weight_map}))
    print(f"[patch] wrote {idx}")


def _load(size_key: str, quant: str):
    _stub_better_transformer()
    # macOS selects AirLLM's MLX persister even though this runner uses the
    # PyTorch Qwen path. Windows/Linux already select the safetensors persister
    # and must not import Apple's unavailable `mlx` package.
    import sys
    if sys.platform == "darwin":
        _patch_mlx_persister()
    _patch_dynamic_cache()
    from airllm.airllm_qwen2 import AirLLMQWen2
    path = str(local_dir(config.MODELS[size_key]["id"]))
    untie_embeddings(path)
    _ensure_shard_index(path)
    compression = config.QUANT_TO_COMPRESSION.get(quant)
    shards_path = config.SHARDS_DIR / size_key
    shards_path.mkdir(parents=True, exist_ok=True)
    model = AirLLMQWen2(path, device="cpu", dtype=torch.float16,
                        compression=compression, hf_token=config.HF_TOKEN or None,
                        layer_shards_saving_path=str(shards_path), max_seq_len=config.MAX_SEQ_LEN)
    tok = model.get_tokenizer(hf_token=config.HF_TOKEN or None)
    return tok, model


def run(size_key: str, quant: str, prompt: str, max_new_tokens: int) -> dict:
    """Run AirLLM (Qwen2.5, CPU path) at a quant level; return a result dict."""
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

    inputs = tok(prompt, return_tensors="pt")
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                      use_cache=False, streamer=streamer, pad_token_id=tok.eos_token_id)

    token_times, output, errors = [], "", []

    def _generate():
        try:
            model.generate(**gen_kwargs)
        except Exception as exc:  # propagate worker failures without deadlocking streamer
            errors.append(exc)
            streamer.on_finalized_text("", stream_end=True)

    with peak_memory() as mt:
        thread = threading.Thread(target=_generate)
        thread.start()
        for piece in streamer:
            token_times.append(now_ms() - t0)
            output += piece
        thread.join()
        rec["peak_rss_mb"] = mt.peak_rss_mb

    if errors:
        exc = errors[0]
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["wall_ms"] = now_ms() - t0
        return rec

    rec["output"] = output
    rec["ok"] = True
    rec["wall_ms"] = now_ms() - t0
    rec["timing"] = summarize_token_times(token_times)
    rec["tokens"] = exact_tokens_with_tokenizer(tok, prompt, output)
    return rec
