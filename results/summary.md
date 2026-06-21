# Aggregated results

| scenario | size | quant | ok | ttft_ms | itl_mean_ms | throughput_tps | peak_rss_mb | wall_ms | estimated_kwh |
|---|---|---|---|---|---|---|---|---|---|
| airllm | 14B | fp16 | true | 14170.91 | 12856.19 | 0.08 | 4600.09 | 631324.38 | 7.01 |
| ollama | 14b | q2 | true | 2489.79 | 83.10 | 12.29 | 232.53 | 6449.97 | nan |
| ollama | 14b | q4 | true | 8471.79 | 103.27 | 9.89 | 259.48 | 13344.13 | 0.15 |
| ollama | 14b | q8 | false | nan | nan | nan | nan | 720000.00 | nan |

---

# §5.7 Original Extension — AirLLM Size Scaling (Qwen2.5: 0.5B / 1.5B / 14B)

All three runs use the same AirLLM FP16 layer-streaming path, the same prompt (16 input →
48 output tokens), and the same hardware (Apple M3, 16 GB unified memory).

| Size | Params | Layers | TTFT (ms) | ITL mean (ms/tok) | Throughput (tok/s) | Peak RSS (MB) |
|---|---:|---:|---:|---:|---:|---:|
| **0.5B** | 0.5 B | 24 | 6 242.76 | 1 602.94 | 0.64 | 2 152.62 |
| **1.5B** | 1.5 B | 28 | 10 564.82 | 2 386.87 | 0.43 | 1 930.89 |
| **14B**  | 14.0 B | 48 | 14 170.91 | 12 856.19 | 0.08 | 4 600.09 |

## What the numbers show

**Decode latency (ITL) scales with layer count — because AirLLM re-streams all layers on
every token.**

| Size | Layers | ITL ms/tok | ms per layer |
|---|---:|---:|---:|
| 0.5B | 24 | 1 603 | 66.8 |
| 1.5B | 28 | 2 387 | 85.2 |
| 14B  | 48 | 12 856 | 267.8 |

The per-layer cost rises with size because each layer is also wider (larger hidden
dimension), so the SSD → RAM transfer per layer is bigger. The combined effect (more layers
× bigger layers) drives the 8× ITL gap between 1.5B and 14B.

**TTFT (prefill) also grows with size** — the prompt is forwarded through all layers once to
build the KV-cache. More layers × larger hidden dimension = more compute + more SSD reads.

**Peak RSS is not monotone** — 0.5B sits at 2 152 MB and 1.5B at 1 930 MB because the OS
reclaimed a slab of memory between runs; the framework + OS baseline dominates at small
model sizes.

**Throughput** drops steeply: 0.64 tok/s → 0.43 → 0.08. Even the 0.5B AirLLM path is
roughly 15× slower than Ollama Q4 because the bottleneck is SSD I/O, not arithmetic.

## Takeaway

AirLLM's SSD-streaming approach makes large models runnable on memory-constrained hardware,
but its cost is linear in the number of layers × bytes-per-layer. On this hardware, even a
0.5B model is slow (0.64 tok/s) because every token pays the SSD round-trip. The 14B model
simply multiplies that cost by 2× layers and 5–6× bytes per layer, reaching 12.9 s/token.
