# EX05 — Technical Report (data-driven)

Subject model: **Qwen/Qwen2.5-14B-Instruct**  
Hardware: Apple M3 MacBook Pro, 16 GB unified memory.  
Generated from `results/` Markdown result files.

## 1. Headline comparison (§5.4)

| Path | Peak RAM (MB) | Throughput (tok/s) | Outcome |
|---|---:|---:|---|
| Baseline (FP16) | — | — | OOM / swap thrash |
| AirLLM (FP16) | 4711 | 0.08 | ran, memory-bound |
| Ollama GGUF (Q4) | 76 | 10.13 | ran, comfortable |

## 2. Aggregated results

| scenario | size | quant | ok | ttft_ms | itl_mean_ms | throughput_tps | peak_rss_mb | wall_ms | estimated_kwh |
|---|---|---|---|---|---|---|---|---|---|
| airllm | 0.5B | fp16 | true | 3711.39 | 1954.98 | 0.52 | 1431.30 | 97600.70 | 1.08 |
| airllm | 1.5B | fp16 | true | 4269.70 | 2421.88 | 0.42 | 1700.45 | 120569.93 | 1.34 |
| airllm | 14B | fp16 | true | 13419.82 | 13021.34 | 0.08 | 4710.98 | 638495.68 | 7.09 |
| baseline | 14B | — | false | — | — | — | — | 93097.58 | 1.03 |
| ollama | 14b | q2 | true | 5903.05 | 82.32 | 12.41 | 259.27 | 9812.34 | 0.11 |
| ollama | 14b | q4 | true | 8828.25 | 100.84 | 10.13 | 75.75 | 13576.96 | 0.15 |
| ollama | 14b | q8 | true | 86119.72 | 17111.69 | 0.06 | 18.08 | 890634.49 | 9.90 |

## 3. Economics — On-Prem vs API (§5.5)

| Path | Latency/req | Break-even vs GPT-4o |
|---|---:|---:|
| Ollama Q4 (~15s) | 15s | 255141 req/mo |
| AirLLM (~600s) | 600s | never |

## 4. Figures

- ![Path comparison](../figures/path_comparison.png)
- ![Quant sweep](../figures/quant_sweep.png)
- ![Size scaling](../figures/size_scaling.png)
- ![Roofline](../figures/roofline.png)
- ![Break-even](../figures/breakeven.png)

## 5. Reproduction

See `../README.md` §7 for full instructions. Raw per-run data is in `results/`.
