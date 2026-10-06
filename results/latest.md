# Benchmarks

GPU: NVIDIA GeForce RTX 5070 Ti, 15.92 GiB, driver 616.92.
TTFT = time to first token; TPOT = time per output token after that. Prompt / output = mean tokens per request as the server counted them (chat template included); target in brackets.

| Server config | Model | Concurrency | Prompt / output tokens | Throughput (tok/s) | TTFT p50 / p95 (ms) | TPOT p50 (ms) | Peak VRAM (GiB) | Peak KV cache | Preemptions | Errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ollama-q4km | qwen2.5:7b-instruct | 1 | 481.8 / 256 [512 / 256] | 118.4 | 110 / 135.3 | 7.92 | 8.11 | – | – | 0/48 |
| ollama-q4km | qwen2.5:7b-instruct | 4 | 482 / 256 [512 / 256] | 112.8 | 6313.3 / 8385.5 | 7.89 | 8.12 | – | – | 0/48 |
| ollama-q4km | qwen2.5:7b-instruct | 16 | 482.9 / 256 [512 / 256] | 99.1 | 37130.9 / 42507.2 | 10.44 | 8.07 | – | – | 0/48 |

Peak KV cache comes from vLLM's `vllm:kv_cache_usage_perc`; current llama.cpp and Ollama don't report it (–).
