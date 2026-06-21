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

# Subject family = Qwen2.5 (GQA). AirLLM 2.11.0's macOS MLX backend hardcodes
# AirLLMLlamaMlx (MHA-only); we bypass it by importing AirLLMQWen2 directly and
# running on CPU (see run_airllm.py). All paths (baseline, AirLLM, Ollama) use
# Qwen2.5. FP16 GB approx.
MODELS = {
    "0.5B": {"id": "Qwen/Qwen2.5-0.5B-Instruct", "params_b": 0.5, "fp16_gb": 1.0,  "layers": 24},
    "1.5B": {"id": "Qwen/Qwen2.5-1.5B-Instruct", "params_b": 1.5, "fp16_gb": 3.0,  "layers": 28},
    "14B":  {"id": "Qwen/Qwen2.5-14B-Instruct",  "params_b": 14.0,"fp16_gb": 28.0, "layers": 48},
}
SUBJECT = "14B"          # AirLLM deep-dive subject (Qwen2.5-14B, GQA via AirLLMQWen2/CPU)
# Size sweep across Qwen2.5. 0.5B ships with tied embeddings; hf_utils.untie_embeddings
# materialises lm_head.weight so AirLLM's splitter accepts it.
SWEEP_SIZES = ["0.5B", "1.5B", "14B"]      # AirLLM size sweep (Qwen2.5, AirLLMQWen2 is Qwen-only)
OLLAMA_SIZES = ["14b"]                   # Ollama/GGUF comparison (Qwen2.5)
QUANT_LEVELS = ["fp16", "q8", "q4", "q2"]
QUANT_TO_COMPRESSION = {"fp16": None, "q8": "8bit", "q4": "4bit", "q2": "2bit"}

PROMPT = (
    "Explain in three short sentences how virtual memory paging works"
    " in modern operating systems."
)
MAX_NEW_TOKENS = 48
MAX_SEQ_LEN = 2048        # KV-cache cap for AirLLM and Ollama (prevents swap on 16 GB)
REPEATS = 1          # single run per scenario; see README §7 for variance note

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
HW_RAM_MB = 16 * 1024          # 16 GB unified memory (CPU + GPU share one pool)
BASELINE_OOM_MB = 20_000.0     # observed MPS peak when baseline OOM'd (~20 GB)
HW_DESCRIPTION = "Apple M3 MacBook Pro, 16 GB unified memory"
MAINT_FRAC_PER_YEAR = 0.02  # 2% of CAPEX/year
# Optional cloud GPU (3rd economics curve, §5.5 optional).
CLOUD_GPU_USD_PER_HOUR = 2.49  # e.g. A10G spot
OLLAMA_BASE_URL = "http://localhost:11434"
