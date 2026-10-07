# Benchmarks

GPU: NVIDIA GeForce RTX 5070 Ti, 15.92 GiB, driver 617.42.
TTFT = time to first token; TPOT = time per output token after that. Prompt / output = mean tokens per request as the server counted them (chat template included); target in brackets.

| Server config | Model | Concurrency | Prompt / output tokens | Throughput (tok/s) | TTFT p50 / p95 (ms) | TPOT p50 (ms) | Peak VRAM (GiB) | Peak KV cache | Preemptions | Errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llamacpp-f16kv-chat | /models/Qwen2.5-7B-Instruct-Q4_K_M.gguf | 1 | 481.8 / 256 [512 / 256] | 128 | 161.2 / 164.4 | 7.22 | 9.28 | – | – | 0/48 |
| llamacpp-f16kv-chat | /models/Qwen2.5-7B-Instruct-Q4_K_M.gguf | 4 | 482 / 256 [512 / 256] | 198.9 | 454.2 / 461.7 | 14.27 | 9.29 | – | – | 0/48 |
| llamacpp-f16kv-chat | /models/Qwen2.5-7B-Instruct-Q4_K_M.gguf | 16 | 482.9 / 256 [512 / 256] | 527.4 | 4363.4 / 4771.7 | 13.56 | 9.29 | – | – | 0/48 |
| llamacpp-f16kv-long | /models/Qwen2.5-7B-Instruct-Q4_K_M.gguf | 8 | 3511.2 / 256 [4096 / 256] | 231.2 | 2096.5 / 2213.2 | 26.75 | 9.34 | – | – | 0/144 |
| llamacpp-f16kv-long | /models/Qwen2.5-7B-Instruct-Q4_K_M.gguf | 16 | 3512 / 256 [4096 / 256] | 225.8 | 9791.4 / 10453.6 | 32.69 | 9.32 | – | – | 0/144 |
| llamacpp-f16kv-long | /models/Qwen2.5-7B-Instruct-Q4_K_M.gguf | 32 | 3512 / 256 [4096 / 256] | 223.2 | 28271.3 / 28530.3 | 33.14 | 9.32 | – | – | 0/144 |
| llamacpp-f16kv-long | /models/Qwen2.5-7B-Instruct-Q4_K_M.gguf | 48 | 3512 / 256 [4096 / 256] | 219.2 | 47318.8 / 48432.6 | 33.55 | 9.5 | – | – | 0/144 |
| llamacpp-q8kv-chat | /models/Qwen2.5-7B-Instruct-Q4_K_M.gguf | 1 | 481.8 / 256 [512 / 256] | 124.5 | 143.3 / 146.3 | 7.5 | 7.68 | – | – | 0/48 |
| llamacpp-q8kv-chat | /models/Qwen2.5-7B-Instruct-Q4_K_M.gguf | 4 | 482 / 256 [512 / 256] | 192.9 | 385.3 / 404.3 | 13.94 | 7.68 | – | – | 0/48 |
| llamacpp-q8kv-chat | /models/Qwen2.5-7B-Instruct-Q4_K_M.gguf | 16 | 482.9 / 256 [512 / 256] | 516.9 | 4373.4 / 4748.3 | 14.12 | 7.68 | – | – | 0/48 |
| llamacpp-q8kv-long | /models/Qwen2.5-7B-Instruct-Q4_K_M.gguf | 8 | 3511.2 / 256 [4096 / 256] | 226.8 | 1707.4 / 2085.3 | 28.62 | 7.83 | – | – | 0/144 |
| llamacpp-q8kv-long | /models/Qwen2.5-7B-Instruct-Q4_K_M.gguf | 16 | 3512 / 256 [4096 / 256] | 216.6 | 10218.8 / 10639.1 | 34.42 | 7.94 | – | – | 0/144 |
| llamacpp-q8kv-long | /models/Qwen2.5-7B-Instruct-Q4_K_M.gguf | 32 | 3512 / 256 [4096 / 256] | 211.7 | 29455 / 30649.5 | 35.05 | 8.03 | – | – | 0/144 |
| llamacpp-q8kv-long | /models/Qwen2.5-7B-Instruct-Q4_K_M.gguf | 48 | 3512 / 256 [4096 / 256] | 202.8 | 50434 / 53126.6 | 36.42 | 7.94 | – | – | 0/144 |
| ollama-q4km | qwen2.5:7b-instruct | 1 | 481.8 / 256 [512 / 256] | 118.4 | 110 / 135.3 | 7.92 | 8.11 | – | – | 0/48 |
| ollama-q4km | qwen2.5:7b-instruct | 4 | 482 / 256 [512 / 256] | 112.8 | 6313.3 / 8385.5 | 7.89 | 8.12 | – | – | 0/48 |
| ollama-q4km | qwen2.5:7b-instruct | 16 | 482.9 / 256 [512 / 256] | 99.1 | 37130.9 / 42507.2 | 10.44 | 8.07 | – | – | 0/48 |
| vllm-fp16kv-chat | Qwen/Qwen2.5-7B-Instruct-AWQ | 1 | 481.8 / 256 [512 / 256] | 114.9 | 144.3 / 148.6 | 8.15 | 15.66 | 0% | 0 | 0/48 |
| vllm-fp16kv-chat | Qwen/Qwen2.5-7B-Instruct-AWQ | 4 | 482 / 256 [512 / 256] | 408.8 | 275.9 / 375.5 | 8.72 | 15.75 | 2% | 0 | 0/48 |
| vllm-fp16kv-chat | Qwen/Qwen2.5-7B-Instruct-AWQ | 16 | 482.9 / 256 [512 / 256] | 1092.4 | 888.8 / 1393.9 | 10.65 | 15.81 | 7% | 0 | 0/48 |
| vllm-fp16kv-long | Qwen/Qwen2.5-7B-Instruct-AWQ | 8 | 3511.2 / 256 [4096 / 256] | 257.2 | 2350 / 2962.6 | 22.26 | 15.81 | 19% | 0 | 0/144 |
| vllm-fp16kv-long | Qwen/Qwen2.5-7B-Instruct-AWQ | 16 | 3512 / 256 [4096 / 256] | 285.6 | 2423.2 / 6295.7 | 46.03 | 15.85 | 38% | 0 | 0/144 |
| vllm-fp16kv-long | Qwen/Qwen2.5-7B-Instruct-AWQ | 32 | 3512 / 256 [4096 / 256] | 294 | 2140.6 / 17341.4 | 101.06 | 15.84 | 76% | 0 | 0/144 |
| vllm-fp16kv-long | Qwen/Qwen2.5-7B-Instruct-AWQ | 48 | 3512 / 256 [4096 / 256] | 296.8 | 5877.1 / 27536.5 | 136.83 | 15.83 | 100% | 6 | 0/144 |
| vllm-fp8kv-chat | Qwen/Qwen2.5-7B-Instruct-AWQ | 1 | 481.8 / 256 [512 / 256] | 120 | 141.4 / 143.2 | 7.82 | 15.27 | 0% | 0 | 0/48 |
| vllm-fp8kv-chat | Qwen/Qwen2.5-7B-Instruct-AWQ | 4 | 482 / 256 [512 / 256] | 437.8 | 265.8 / 353.8 | 8.11 | 15.42 | 1% | 0 | 0/48 |
| vllm-fp8kv-chat | Qwen/Qwen2.5-7B-Instruct-AWQ | 16 | 482.9 / 256 [512 / 256] | 1209.9 | 865.2 / 1294.8 | 9.81 | 15.57 | 4% | 0 | 0/48 |
| vllm-fp8kv-long | Qwen/Qwen2.5-7B-Instruct-AWQ | 8 | 3511.2 / 256 [4096 / 256] | 296.2 | 2179.6 / 2938.1 | 18.12 | 15.59 | 11% | 0 | 0/144 |
| vllm-fp8kv-long | Qwen/Qwen2.5-7B-Instruct-AWQ | 16 | 3512 / 256 [4096 / 256] | 326.9 | 2546.8 / 5828.3 | 39.11 | 15.58 | 22% | 0 | 0/144 |
| vllm-fp8kv-long | Qwen/Qwen2.5-7B-Instruct-AWQ | 32 | 3512 / 256 [4096 / 256] | 350.7 | 1824.4 / 15628.1 | 84 | 15.57 | 44% | 0 | 0/144 |
| vllm-fp8kv-long | Qwen/Qwen2.5-7B-Instruct-AWQ | 48 | 3512 / 256 [4096 / 256] | 359 | 1840 / 25851.8 | 125.66 | 15.58 | 65% | 0 | 0/144 |

Peak KV cache comes from vLLM's `vllm:kv_cache_usage_perc`; current llama.cpp and Ollama don't report it (–).

## KV cache vLLM allocated

| Server config | KV dtype | GPU blocks | Tokens of KV cache |
| --- | --- | ---: | ---: |
| vllm-fp16kv-chat | auto | 9,878 | 158,048 |
| vllm-fp16kv-long | auto | 9,878 | 158,048 |
| vllm-fp8kv-chat | fp8 | 17,019 | 272,304 |
| vllm-fp8kv-long | fp8 | 17,019 | 272,304 |

Compare with `inference-lab plan vllm` for the same model, context and memory settings.
