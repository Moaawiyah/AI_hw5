# EX05 — Running a Massive LLM Locally: AirLLM, Quantization & Performance Benchmarking

A deep-dive technical report on running an LLM that is **too large for the hardware** on a
consumer MacBook, documenting the inevitable bottleneck, and applying two optimization
techniques — **AirLLM** (layer-streaming / OS-paging analogy) and **GGUF quantization** —
then benchmarking, drawing a **Roofline**, and performing a full **On-Prem vs API
economics** analysis with a break-even point.

> Author: _<your name>_ · Course: L08, Assignment 05 (Dr. Yoram Segal) · Tooling: `uv`, Python 3.12

---

## 1. Hardware & Model Justification (§5.1)

| Component | Specification |
|---|---|
| Machine | Apple MacBook Pro (Mac15,3) |
| SoC | Apple M3 — 8 cores (4 Performance + 4 Efficiency) |
| Memory | **16 GB unified memory** (CPU + GPU share one pool — there is **no separate VRAM**) |
| GPU | Metal 4 |
| Storage | 460 GB NVMe SSD (~225 GB free for this experiment) |

**The single most important fact:** Apple Silicon uses **unified memory**. The
assignment's classic "RAM vs VRAM" split collapses into one 16 GB budget. Every byte the
model occupies competes with the OS, the KV-cache, and the framework itself.

**Model chosen — Llama-2-13B-chat (FP16 ≈ 26 GB).** It is deliberately **~1.6× larger
than the entire 16 GB unified pool**, so a naive direct load *must* fail (§5.2). It is the
"big, but not too big" sweet-spot the brief asks for: too big to run directly, small enough
that AirLLM can still stream it from the SSD.

**Why Llama-2 and not Qwen2.5?** This is itself a finding. AirLLM 2.11.0's macOS backend is
hard-wired to `AirLLMLlamaMlx`, which (a) assumes **standard multi-head attention (MHA)**,
and (b) expects **Llama-style layer weights**. Every modern GQA model — Llama-3.x, Qwen2.5,
Mistral — mis-reshapes on the MLX path (`ValueError: [reshape] ...`). Llama-2 (classic MHA)
is the one family AirLLM actually supports on Apple Silicon. Qwen2.5-14B is therefore kept
as the **Ollama/GGUF second path** (which has no such restriction).

---

## 2. Experiment Design & Measurement Tools

Three deployment paths, all measured on the **same prompt** (18 input tokens, 48 new tokens)
with identical methodology:

| Path | Tool | Format | Purpose |
|---|---|---|---|
| **Baseline** | `transformers` + MPS | FP16 safetensors | Naive direct run — expected to fail (§5.2) |
| **AirLLM** | `airllm.AutoModel` (MLX) | FP16 layer shards | Layer-streaming optimization (§5.3) |
| **Quantized** | Ollama | GGUF (q2/q4/q8) | Quantization sweep (§5.3) |

**Metrics captured per run** (`src/metrics.py`): **TTFT** (Time-To-First-Token — Prefill /
compute-bound), **ITL/TPOT** (Inter-Token-Latency — Decode / memory-bound), **throughput**
(tok/s), **peak RSS** (sampled in a background thread), wall-time, and input/output token
counts. Raw numbers are persisted as JSON in `results/` (§6.1 — keep all raw numbers).

> **Note on quantization.** AirLLM's built-in `compression=4bit/8bit` relies on
> `bitsandbytes`, which is **CUDA-only**. On Apple Silicon the quantization study is
> therefore performed with **GGUF via Ollama** (an assignment keyword). This cleanly
> separates the two techniques: AirLLM = the paging/memory technique, GGUF = the
> quantization technique.

---

## 3. Findings

### 3.1 Baseline — the failure (§5.2)

```
RuntimeError: MPS backend out of memory
(MPS allocated: 19.98 GiB, max allowed: 20.13 GiB)
```
The 13B FP16 weights (~26 GB) cannot fit. MPS exploited ~4 GB of swap headroom, reached
~20 GB, and OOM'd at 79 % of weight loading. **The bottleneck is unambiguously memory
(VRAM/unified RAM), not compute** — answering Research Question 1.

### 3.2 Headline comparison — the three paths

![Path comparison](figures/path_comparison.png)

| Path | Peak RAM | Throughput | Outcome |
|---|---|---|---|
| Baseline (13B FP16) | ~20 000 MB (OOM) | 0 tok/s | **Cannot run** |
| **AirLLM (13B FP16)** | **2 429 MB** | 0.09 tok/s | Runs in ~⅓ the RAM budget |
| Ollama GGUF (Qwen2.5-14B Q4) | 232 MB | 9.7 tok/s | Runs comfortably |

**The AirLLM story in one number:** a **26 GB** model runs in **2.4 GB** of RAM. The
trade-off is speed — **0.09 tok/s (11.4 s/token)** — because every generated token forces
the system to re-stream all 40 transformer layers from the SSD via `mmap`.

### 3.3 AirLLM resource shift (§5.3)

AirLLM shards the model into per-layer `.mlx` files (`layer_shards_saving_path`, set
per-model per §6.1) and loads/evicts one layer at a time. Memory is traded for **disk
I/O**: the binding constraint moves from *VRAM capacity* to *SSD bandwidth*. This is
exactly **OS virtual-memory paging** — AirLLM is `mmap` + on-demand layer paging for
neural networks (see §5).

### 3.4 Quantization sweep — GGUF (§5.3 / §5.4)

![Quantization sweep](figures/quant_sweep.png)

| GGUF quant | Disk size | TTFT | ITL | Throughput | Quality |
|---|---|---|---|---|---|
| Q2_K | 5.8 GB | 2.5 s | 83 ms | **12.3 tok/s** | slightly degraded but on-topic |
| Q4_K_M | 9.0 GB | 8.8 s | 105 ms | 9.7 tok/s | faithful |
| Q8_0 | 15 GB | — | — | **swap-thrash** | unusable (fills 16 GB) |

**The "red line" of accuracy (Research Q3):** Q4_K_M is the practical sweet-spot (faithful,
fits RAM). Q2_K stays coherent for simple factual prompts but loses nuance. **Q8_0 is the
memory cliff** — at 15 GB it saturates the 16 GB pool and re-introduces the same swap
bottleneck as the FP16 baseline. Quantization defers the memory wall; it does not remove it.

### 3.5 Prefill vs Decode (Research Q4)

| Stage | Metric | What it measures | Our data (AirLLM 13B) |
|---|---|---|---|
| **Prefill** | TTFT | compute (build KV-cache) | 92.9 s |
| **Decode** | ITL/TPOT | memory-bandwidth (stream weights) | 11.4 s/token |

TTFT (Prefill) and ITL (Decode) cleanly separate the two regimes. The grotesque ITL proves
the system is **memory-bandwidth-bound** during generation — every token waits on the SSD.

---

## 4. Original Extension (§5.7): Model-Size Scaling

![Size scaling](figures/size_scaling.png)

We swept AirLLM FP16 across two Llama-2 sizes to see how Prefill (TTFT) and Decode (ITL)
scale with model size:

| Model | Layers | ITL (ms/tok) | TTFT (s) |
|---|---|---|---|
| Llama-2-7B | 32 | 6 480 | 7.4 |
| Llama-2-13B | 40 | 11 380 | 92.9 |

Decode latency grows roughly **linearly with layer count** (each token must re-stream every
layer), confirming the memory-bandwidth-bound scaling law.

---

## 5. Linking Results to Lecture Concepts (§5.6)

- **VRAM** — On this machine VRAM *is* the 16 GB unified pool; there is no separate GPU
  memory. The OOM at ~20 GB is the unified-memory wall.
- **Prefill vs Decode** — TTFT isolates the compute-heavy Prefill; ITL isolates the
  memory-heavy Decode. Our ~11 s/token Decode is the signature of memory-boundedness.
- **compute-bound vs memory-bound** — see the **Roofline** below. Decode sits far to the
  left of the ridgepoint (low arithmetic intensity → memory-bound); Prefill is further right.
- **Virtual memory & Paging** — AirLLM *is* demand paging for weights: layers are pages,
  the SSD is the backing store, `mmap` is the fault handler. The analogy is exact.

![Roofline](figures/roofline.png)

---

## 6. Economics: On-Prem vs API (§5.5)

![Break-even](figures/breakeven.png)

**Assumptions (all stated for reproducibility):** CAPEX = $1 999 (M3 MBP), lifetime 4 yr,
electricity $0.30/kWh, load 40 W, maintenance 2 % CAPEX/yr. API = GPT-4o list prices
($2.50 / $10.00 per 1 M in/out tokens), with an 80 %-prompt-cache variant. ~18 input +
50 output tokens per request.

| On-Prem path | Latency/req | Electricity/req | **Break-even vs GPT-4o** |
|---|---|---|---|
| **Ollama Q4** | ~15 s | $0.00005 | **≈ 255 k requests/month** |
| AirLLM (13B) | ~600 s | $0.002 | **never** (electricity alone > API) |

**Recommendation (Research Q6):**
- **Below ~250 k req/month** → use the **API** (cheaper, faster, zero maintenance).
- **Above ~250 k req/month, latency-tolerant, and Q4 quality acceptable** → **Ollama
  on-prem** becomes cheaper.
- **AirLLM** is **never** competitive on raw cost — its value is enabling inference of an
  *otherwise unrunnable* model when no API exists (privacy/offline/closed-data scenarios),
  not saving money.
- **Privacy/security:** on-prem (either path) wins whenever data cannot leave the machine —
  regardless of cost. Prompt-caching (PageAttention-style, §5.5) shifts the API curve down
  and pushes the break-even further out.

---

## 7. How to Reproduce

```bash
# 1. Environment (Python 3.12 — the brief warns against the newest Python)
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Set HF token (never hard-code it — §6.2)
cp .env.example .env  # then edit .env

# 3. Download models (Llama-2 chat family for AirLLM; Ollama pulls GGUF separately)
python -m src.hf_utils 7B 13B
ollama pull qwen2.5:14b qwen2.5:14b-instruct-q2_K qwen2.5:14b-instruct-q8_0

# 4. Run experiments (each persists JSON to results/)
python -m src.bench_driver baseline --size 13B --repeats 1   # expected OOM
python -m src.bench_driver airllm   --size 13B --quants fp16 --repeats 1
python -m src.bench_driver airllm   --size 7B  --quants fp16 --repeats 1   # sweep
python -m src.bench_driver ollama   --size 14b --repeats 1                  # Q4 default

# 5. Generate figures + tables
python -m src.analyze_perf        # path_comparison.png, quant_sweep.png, summary.csv
python -m src.analyze_scaling     # size_scaling.png
python -m src.analyze_roofline    # roofline.png
python -m src.plot_economics      # breakeven.png
```

**Repository structure** (§9): every Python file is **≤ 150 lines of code**.

```
src/        config, hf_utils, token_count, metrics,
            run_baseline, run_airllm, run_ollama, run_sweep, bench_driver,
            analyze_perf, analyze_scaling, analyze_roofline, economics, plot_economics
experiments/prompts.py
results/    raw JSON + summary.csv
figures/    all embedded charts
models_cache/  (git-ignored)
```

---

## 8. Negative results & lessons (the AirLLM-on-Apple-Silicon journey)

The brief explicitly values well-analyzed negative results. We document the four
AirLLM↔Apple-Silicon incompatibilities we diagnosed and overcame, each a real finding:

1. **`bias` parameter rejection** — modern `mlx.nn.Module.update` is `strict=True` by
   default; airlll calls it without `strict=False`. Patched via a one-line monkeypatch.
2. **GQA reshape failure** — airlll's MLX backend assumes MHA; Llama-3.x/Qwen2.5 break.
   Resolved by selecting Llama-2 (MHA).
3. **Cross-model shard pollution** — a shared shard dir collided between models; fixed with
   per-model `layer_shards_saving_path` (§6.1).
4. **Phantom `.bin` shard index** — the downloaded `pytorch_model.bin.index.json` made
   airlll look for 3 non-existent `.bin` files; resolved by excluding `.bin` from downloads.

Each is captured in the commit history and the runner code.
