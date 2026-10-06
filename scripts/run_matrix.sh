#!/usr/bin/env bash
# Benchmark every server config in turn, one on the GPU at a time, then build results/latest.md.
#   scripts/run_matrix.sh                 # vLLM fp16/fp8 KV, llama.cpp f16/q8_0 KV, Ollama
#   CONC="1 4 16 32" REQ=64 scripts/run_matrix.sh
#   OFFLOAD=1 scripts/run_matrix.sh       # adds Qwen2.5-32B with partial GPU offload (run get_models.sh first)
set -uo pipefail
cd "$(dirname "$0")/.."
CONC="${CONC:-1 4 16}"
REQ="${REQ:-32}"

wait_ready() {  # $1 = base URL; first vLLM start downloads the model, so allow 20 minutes
  for _ in $(seq 1 240); do
    curl -sf "$1/models" >/dev/null && return 0
    sleep 5
  done
  return 1
}

run() {  # run <label> <profile> <base url> [VAR=value ...]
  local label=$1 profile=$2 url=$3; shift 3
  echo "=== $label ==="
  env "$@" docker compose --profile "$profile" up -d
  if wait_ready "$url"; then
    [ "$profile" = ollama ] && docker compose exec ollama ollama pull qwen2.5:7b-instruct
    inference-lab bench --backend "$profile" --label "$label" --concurrency $CONC --requests "$REQ" \
      || echo "!!! $label: benchmark failed"
  else
    echo "!!! $label: server never became ready"; docker compose --profile "$profile" logs --tail 30
  fi
  docker compose --profile "$profile" down
}

run vllm-fp16kv     vllm     http://localhost:8000/v1  KV_DTYPE=auto
run vllm-fp8kv      vllm     http://localhost:8000/v1  KV_DTYPE=fp8
run llamacpp-f16kv  llamacpp http://localhost:8081/v1  CACHE_K=f16 CACHE_V=f16
run llamacpp-q8kv   llamacpp http://localhost:8081/v1  CACHE_K=q8_0 CACHE_V=q8_0
run ollama-q4km     ollama   http://localhost:11434/v1
if [ "${OFFLOAD:-0}" = 1 ]; then
  NGL=$(python -c "
from inference_lab.models import get_model
from inference_lab.planner import plan_llamacpp
import os
gib = os.path.getsize('models/Qwen2.5-32B-Instruct-Q4_K_M.gguf') / 1024**3
print(plan_llamacpp(get_model('qwen2.5-32b'), gib, 8192, 16, 'q8_0', 'q8_0').gpu_layers)")
  echo "planner says -ngl $NGL for the 32B model"
  CONC="1 2" run "llamacpp-32b-ngl$NGL" llamacpp http://localhost:8081/v1 \
    GGUF_FILE=Qwen2.5-32B-Instruct-Q4_K_M.gguf NGL="$NGL" CTX=8192 PARALLEL=2 CACHE_K=q8_0 CACHE_V=q8_0
fi
inference-lab report
