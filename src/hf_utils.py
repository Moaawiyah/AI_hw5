"""Hugging Face helpers: token loading + robust snapshot download."""
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
