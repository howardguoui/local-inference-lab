#!/usr/bin/env bash
# Benchmark every server config in turn, one on the GPU at a time, then build results/latest.md.
# Each config runs two scenarios:
#   chat  512-token prompts, 256 output tokens, concurrency 1 4 16
#   long  4096-token prompts, 256 output tokens, concurrency 8 16 32 48, 144 requests per level:
#         48 in flight x ~4.4k tokens is ~210k tokens of KV cache, more than vLLM's FP16 cache holds
#
#   scripts/run_matrix.sh
#   OFFLOAD=1 scripts/run_matrix.sh   # adds Qwen2.5-32B with partial GPU offload (run get_models.sh first)
#
# Note: llama.cpp and Ollama run 8 parallel slots (PARALLEL); at higher concurrency their extra
# requests queue, which shows up as TTFT. vLLM batches up to 64 sequences (MAX_NUM_SEQS).
set -uo pipefail
cd "$(dirname "$0")/.."
REQ="${REQ:-48}"        # chat: requests per level
LONG_REQ="${LONG_REQ:-144}"  # long: 3 waves at the top concurrency

wait_ready() {  # $1 = base URL; the first vLLM start downloads the model, so allow 20 minutes
  for _ in $(seq 1 240); do
    curl -sf "$1/models" >/dev/null && return 0
    sleep 5
  done
  return 1
}

bench() {  # bench <label> <backend> <prompt tokens> <requests per level> <concurrency...>
  local label=$1 backend=$2 prompt=$3 requests=$4; shift 4
  inference-lab bench --backend "$backend" --label "$label" --prompt-tokens "$prompt" --max-tokens 256 \
    --requests "$requests" --concurrency "$@" || echo "!!! $label: benchmark failed"
}

run() {  # run <label> <profile> <base url> [VAR=value ...]
  local label=$1 profile=$2 url=$3; shift 3
  echo "=== $label ==="
  env "$@" docker compose --profile "$profile" up -d
  if wait_ready "$url"; then
    if [ "$profile" = ollama ]; then
      docker compose --profile ollama exec ollama ollama pull qwen2.5:7b-instruct
    fi
    bench "$label-chat" "$profile" 512 "$REQ" 1 4 16
    bench "$label-long" "$profile" 4096 "$LONG_REQ" 8 16 32 48
  else
    echo "!!! $label: server never became ready"
    docker compose --profile "$profile" logs --tail 30
  fi
  docker compose --profile "$profile" down
}

run vllm-fp16kv     vllm     http://localhost:8000/v1  KV_DTYPE=auto
run vllm-fp8kv      vllm     http://localhost:8000/v1  KV_DTYPE=fp8
run llamacpp-f16kv  llamacpp http://localhost:8081/v1  CACHE_K=f16 CACHE_V=f16
run llamacpp-q8kv   llamacpp http://localhost:8081/v1  CACHE_K=q8_0 CACHE_V=q8_0
run ollama-q4km     ollama   http://localhost:11434/v1

if [ "${OFFLOAD:-0}" = 1 ]; then
  GGUF=models/Qwen2.5-32B-Instruct-Q4_K_M.gguf
  # plan against the VRAM free right now, with the same context and cache types the server will use
  NGL=$(inference-lab plan llamacpp --gguf-path "$GGUF" --ctx 8192 -ctk q8_0 -ctv q8_0 | awk '$1 == "ngl" {print $2}')
  if [ -z "$NGL" ] || [ "$NGL" = 0 ]; then
    echo "!!! could not plan -ngl for $GGUF; skipping the offload run"
  else
    echo "planner: -ngl $NGL for the 32B model"
    env GGUF_FILE="$(basename "$GGUF")" NGL="$NGL" CTX=8192 PARALLEL=2 CACHE_K=q8_0 CACHE_V=q8_0 \
      docker compose --profile llamacpp up -d
    if wait_ready http://localhost:8081/v1; then
      bench "llamacpp-32b-ngl$NGL" llamacpp 512 8 1 2
    fi
    docker compose --profile llamacpp down
  fi
fi
inference-lab report
