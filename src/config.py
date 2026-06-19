"""Central configuration: paths, model registry, prompt, token caps, economics assumptions."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_CACHE = ROOT / "models_cache"
SHARDS_DIR = ROOT / "models_cache" / "airllm_shards"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

for _d in (MODELS_CACHE, SHARDS_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Subject family = Llama-2 (MHA): the only architecture airllm 2.11.0's macOS MLX
# backend supports. Modern GQA models (Llama-3.x, Qwen2.5) mis-reshape on the MLX
# path. We use the -chat (instruct) variants for meaningful output-quality eval (§5.4).
# Qwen2.5-14B is kept as the Ollama/GGUF second-path comparison. FP16 GB approx.
MODELS = {
    # Llama-2 chat family (AirLLM subject + sweep) — MHA, airllm-compatible
    "7B":  {"id": "NousResearch/Llama-2-7b-chat-hf",  "params_b": 7.0, "fp16_gb": 13.0, "layers": 32},
    "13B": {"id": "NousResearch/Llama-2-13b-chat-hf", "params_b": 13.0,"fp16_gb": 26.0, "layers": 40},
    # Qwen2.5 (Ollama/GGUF second path)
    "0.5B": {"id": "Qwen/Qwen2.5-0.5B-Instruct", "params_b": 0.5, "fp16_gb": 1.0,  "layers": 24},
    "14B":  {"id": "Qwen/Qwen2.5-14B-Instruct",  "params_b": 14.0,"fp16_gb": 28.0, "layers": 40},
}
SUBJECT = "13B"                    # AirLLM deep-dive subject (Llama-2-13B, MHA)
SWEEP_SIZES = ["7B", "13B"]        # AirLLM size sweep (Llama-2)
OLLAMA_SIZES = ["14b"]             # Ollama/GGUF comparison (Qwen2.5)
QUANT_LEVELS = ["fp16", "q8", "q4", "q2"]
QUANT_TO_COMPRESSION = {"fp16": None, "q8": "8bit", "q4": "4bit", "q2": "2bit"}

PROMPT = "Explain in three short sentences how virtual memory paging works in modern operating systems."
MAX_NEW_TOKENS = 48
REPEATS = 2

# --- Economics assumptions (state ALL explicitly in report; §5.5) ---
# API pricing (USD per 1M tokens), public list prices as of mid-2026.
API_PRICES = {
    "gpt-4o":       {"input": 2.50, "output": 10.00, "cache_input": 1.25},
    "claude-sonnet":{"input": 3.00, "output": 15.00, "cache_input": 0.30},
}
# On-prem: M3 MacBook Pro 14" 16GB.
HW_CAPEX_USD = 1999.0
HW_LIFETIME_YEARS = 4
ELECTRICITY_KWH_USD = 0.30
HW_IDLE_W = 7
HW_LOAD_W = 40
MAINT_FRAC_PER_YEAR = 0.02  # 2% of CAPEX/year
# Optional cloud GPU (3rd economics curve, §5.5 optional).
CLOUD_GPU_USD_PER_HOUR = 2.49  # e.g. A10G spot
