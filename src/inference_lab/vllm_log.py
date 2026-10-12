"""Read vLLM's startup log: what it measured while sizing the KV cache.

The planner predicts the KV cache from `gpu_memory_utilization x VRAM - weights - overhead`. vLLM does the same
sum with measured numbers and logs each one (current source, vllm/v1/worker/ and vllm/v1/core/kv_cache_utils.py):

    Model loading took 5.19 GiB memory and 12.345678 seconds          gpu_model_runner.py
    Memory profiling takes 3.21 seconds. Total non KV cache memory:    utils/mem_utils.py (debug level)
      6.88GiB; torch peak memory increase: ...; weights memory: 5.19GiB.
    Estimated CUDA graph memory: 0.48 GiB total                        gpu_model_runner.py
    Available KV cache memory: 7.31 GiB                                 gpu_worker.py
    GPU KV cache size: 136,864 tokens                                   kv_cache_utils.py
      Maximum concurrency for 8,192 tokens per request: 16.71x
    Graph capturing finished in 9 secs, took 0.45 GiB                  gpu_model_runner.py
    CUDA graph pool memory: 0.45 GiB (actual), 0.48 GiB (estimated),   gpu_worker.py
      difference: 0.03 GiB (6.7%).

vLLM rounds GiB figures to two decimals. Lines may carry a process prefix ("(EngineCore_DP0 pid=7) INFO ...")
and ANSI colours; both are ignored. If a log holds several startups, the last one wins.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_NUM = r"([0-9][0-9,]*(?:\.[0-9]+)?)"

PATTERNS = {
    "weights_gib": re.compile(rf"Model loading took {_NUM} ?GiB"),
    "non_kv_gib": re.compile(rf"Total non KV cache memory: {_NUM} ?GiB"),
    "kv_available_gib": re.compile(rf"Available KV cache memory: {_NUM} ?GiB"),
    "kv_cache_tokens": re.compile(rf"KV cache size: {_NUM} tokens"),
    "max_model_len": re.compile(rf"Maximum concurrency for {_NUM} tokens per request"),
    "max_concurrency": re.compile(rf"Maximum concurrency for [0-9,]+ tokens per request: {_NUM}x"),
    "cuda_graph_estimate_gib": re.compile(rf"Estimated CUDA graph memory: {_NUM} ?GiB"),
    "cuda_graph_gib": re.compile(rf"Graph capturing finished in [0-9.]+ secs?, took {_NUM} ?GiB"),
}
INT_FIELDS = {"kv_cache_tokens", "max_model_len"}


@dataclass
class VllmStartup:
    kv_available_gib: float
    kv_cache_tokens: int | None = None
    max_model_len: int | None = None
    max_concurrency: float | None = None
    weights_gib: float | None = None
    non_kv_gib: float | None = None
    cuda_graph_estimate_gib: float | None = None
    cuda_graph_gib: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def parse_vllm_log(text: str) -> VllmStartup:
    """Memory figures from a vLLM startup log. Raises ValueError without an "Available KV cache memory" line,
    which every successful vLLM V1 startup prints."""
    text = _ANSI.sub("", text)
    found: dict[str, float | int] = {}
    for field, pattern in PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            raw = matches[-1].replace(",", "")
            found[field] = int(float(raw)) if field in INT_FIELDS else float(raw)
    if "kv_available_gib" not in found:
        raise ValueError(
            "No 'Available KV cache memory' line in this log. Pass the log of a vLLM server that finished "
            "starting, e.g. `docker compose logs vllm > vllm.log`."
        )
    return VllmStartup(**found)


def read_vllm_log(path: str | Path) -> VllmStartup:
    return parse_vllm_log(Path(path).read_text(encoding="utf-8", errors="replace"))
