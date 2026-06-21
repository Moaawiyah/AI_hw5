# Aggregated results

| scenario | size | quant | ok | ttft_ms | itl_mean_ms | throughput_tps | peak_rss_mb | wall_ms | estimated_kwh |
|---|---|---|---|---|---|---|---|---|---|
| airllm | 0.5B | fp16 | true | 3711.39 | 1954.98 | 0.52 | 1431.30 | 97600.70 | 1.08 |
| airllm | 1.5B | fp16 | true | 4269.70 | 2421.88 | 0.42 | 1700.45 | 120569.93 | 1.34 |
| airllm | 14B | fp16 | true | 13419.82 | 13021.34 | 0.08 | 4710.98 | 638495.68 | 7.09 |
| baseline | 14B | â€” | false | — | — | — | — | 93097.58 | 1.03 |
| ollama | 14b | q2 | true | 5903.05 | 82.32 | 12.41 | 259.27 | 9812.34 | 0.11 |
| ollama | 14b | q4 | true | 8828.25 | 100.84 | 10.13 | 75.75 | 13576.96 | 0.15 |
| ollama | 14b | q8 | true | 86119.72 | 17111.69 | 0.06 | 18.08 | 890634.49 | 9.90 |
