#!/usr/bin/env bash
# Download the GGUF files llama.cpp serves into ./models. vLLM and Ollama fetch their own.
#   scripts/get_models.sh            # Qwen2.5-7B-Instruct Q4_K_M (4.7 GB)
#   OFFLOAD=1 scripts/get_models.sh  # also Qwen2.5-32B-Instruct Q4_K_M (~20 GB), for the partial-offload run
set -euo pipefail
cd "$(dirname "$0")/.."
python - <<PY
import os
from huggingface_hub import hf_hub_download

files = [("bartowski/Qwen2.5-7B-Instruct-GGUF", "Qwen2.5-7B-Instruct-Q4_K_M.gguf")]
if os.environ.get("OFFLOAD") == "1":
    files.append(("bartowski/Qwen2.5-32B-Instruct-GGUF", "Qwen2.5-32B-Instruct-Q4_K_M.gguf"))
for repo, name in files:
    print("downloading", name)
    hf_hub_download(repo, name, local_dir="models")
PY
