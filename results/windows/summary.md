# Windows benchmark — Qwen2.5

Hardware: ASUS TUF Gaming F15, i7-13700H, 23.6 GB RAM, RTX 4060 Laptop GPU (8 GB).

| Path | TTFT | ITL | Throughput | Wall | Outcome |
|---|---:|---:|---:|---:|---|
| Baseline 14B FP16 | — | — | — | 32.18 s | CUDA OOM |
| AirLLM 14B FP16 | 123.07 s | 53.01 s | 0.02 tok/s | 44.46 min | ran |
| Ollama 14B Q2_K (warm mean) | 5.07 s | 32.94 ms | 31.00 tok/s | 6.64 s | ran |
| Ollama 14B Q4_K_M (warm mean) | 5.24 s | 111.88 ms | 9.14 tok/s | 10.52 s | ran |
| Ollama 14B Q8_0 (warm mean) | 5.59 s | 378.88 ms | 2.70 tok/s | 23.43 s | ran |

## AirLLM size scaling

| Size | Peak RSS | TTFT | ITL | Throughput | Wall |
|---|---:|---:|---:|---:|---:|
| 0.5B | 2410 MB | 8.03 s | 4.93 s | 0.21 tok/s | 4.08 min |
| 1.5B | 2809 MB | 27.51 s | 7.66 s | 0.13 tok/s | 6.58 min |
| 14B | 7084 MB | 123.07 s | 53.01 s | 0.02 tok/s | 44.46 min |
