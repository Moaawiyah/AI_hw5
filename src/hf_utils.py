"""Hugging Face helpers: token loading + robust snapshot download."""
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

from src import config


def get_token() -> str:
    """Return HF token from env, or empty string. Never log it."""
    return os.environ.get("HF_TOKEN", "")


def local_dir(model_id: str) -> Path:
    """Stable local folder for a model id, e.g. models_cache/Qwen2.5-14B-Instruct."""
    name = model_id.split("/")[-1]
    return config.MODELS_CACHE / name


def untie_embeddings(model_dir) -> None:
    """Add lm_head.weight to single-shard Qwen2.5 variants that ship with
    `tie_word_embeddings=true` (0.5B/1.5B/3B). AirLLM's splitter hard-fails
    (`IndexError: shards[0]`) when lm_head is absent, so we materialise the
    tied weight (= a clone of model.embed_tokens.weight — exactly what the
    "tied" in tie_word_embeddings means) and flip the config flag. The
    generated logits are bit-identical to the original tied model.

    Idempotent: skips models that are already untied or already patched."""
    model_dir = Path(model_dir)
    cfg_path, single = model_dir / "config.json", model_dir / "model.safetensors"
    if not (cfg_path.exists() and single.exists()):
        return
    cfg = json.loads(cfg_path.read_text())
    if not cfg.get("tie_word_embeddings", False):
        return
    from safetensors.torch import load_file, save_file
    state = load_file(str(single))
    if "lm_head.weight" in state:
        return
    print(f"[patch] untying embeddings for {model_dir.name}: "
          f"cloning embed_tokens.weight -> lm_head.weight")
    state["lm_head.weight"] = state["model.embed_tokens.weight"].clone()
    save_file(state, str(single), metadata={"format": "pt"})
    cfg["tie_word_embeddings"] = False
    cfg_path.write_text(json.dumps(cfg, indent=2))
    # Stale index/shards would now miss lm_head — drop them so airllm rebuilds.
    for stale in (model_dir / "model.safetensors.index.json",
                  config.SHARDS_DIR / model_dir.name):
        if stale.is_dir():
            import shutil
            shutil.rmtree(stale)
        elif stale.exists():
            stale.unlink()


def download_model(size_key: str) -> Path:
    """Download one model from the registry by size key (e.g. '14B')."""
    if size_key not in config.MODELS:
        raise KeyError(f"unknown size {size_key!r}; choose from {list(config.MODELS)}")
    model_id = config.MODELS[size_key]["id"]
    target = local_dir(model_id)
    token = get_token()
    print(f"[download] {model_id} -> {target}")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(target),
        token=token or None,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "vocab*", "merges*"],
        ignore_patterns=["pytorch_model*.bin*", "*.pth", "*.gguf"],
        resume_download=True,
    )
    print(f"[download] done: {target}")
    return target


def download_family(sizes=None) -> dict:
    """Download the whole (or selected) size family. Returns {size: path}."""
    sizes = sizes or config.SWEEP_SIZES
    out = {}
    for s in sizes:
        out[s] = str(download_model(s))
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("sizes", nargs="*", default=None, help="size keys, e.g. 0.5B 14B")
    args = p.parse_args()
    download_family(args.sizes or None)
