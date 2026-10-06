"""Read the Prometheus /metrics endpoint that vLLM and llama.cpp (with --metrics) expose.

The numbers that matter for capacity are the KV cache fill level and, for vLLM,
preemptions: when the cache is full vLLM evicts running sequences and recomputes
them later, which shows up as latency spikes. Both are read here, along with the
cache allocation vLLM actually made (cache_config_info), so the planner's
prediction can be checked against reality.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+(\S+)")
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')

# Fraction 0..1. vLLM renamed gpu_cache_usage_perc to kv_cache_usage_perc in its V1 engine.
KV_USAGE = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc", "llamacpp:kv_cache_usage_ratio")
RUNNING = ("vllm:num_requests_running", "llamacpp:requests_processing")
WAITING = ("vllm:num_requests_waiting", "llamacpp:requests_deferred")
PREEMPTIONS = ("vllm:num_preemptions_total", "vllm:num_preemptions")


@dataclass
class Sample:
    name: str
    value: float
    labels: dict[str, str] = field(default_factory=dict)


def parse(text: str) -> list[Sample]:
    out = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name, raw_labels, raw_value = m.groups()
        try:
            value = float(raw_value)
        except ValueError:
            continue
        labels = {k: v.replace('\\"', '"').replace("\\\\", "\\") for k, v in _LABEL.findall(raw_labels or "")}
        out.append(Sample(name, value, labels))
    return out


def _values(samples: list[Sample], names: tuple[str, ...]) -> list[float]:
    for name in names:  # first name present wins (newest naming first)
        vals = [s.value for s in samples if s.name == name and not math.isnan(s.value)]
        if vals:
            return vals
    return []


def summarize(samples: list[Sample]) -> dict:
    kv = _values(samples, KV_USAGE)
    running = _values(samples, RUNNING)
    waiting = _values(samples, WAITING)
    preempt = _values(samples, PREEMPTIONS)
    hits = _values(samples, ("vllm:prefix_cache_hits_total",))
    queries = _values(samples, ("vllm:prefix_cache_queries_total",))
    out: dict = {
        "kv_cache_usage": max(kv) if kv else None,  # max across engines
        "requests_running": sum(running) if running else None,
        "requests_waiting": sum(waiting) if waiting else None,
        "preemptions_total": sum(preempt) if preempt else None,
        "prefix_cache_hit_rate": (sum(hits) / sum(queries)) if hits and queries and sum(queries) else None,
    }
    info = next((s.labels for s in samples if s.name == "vllm:cache_config_info"), None)
    if info:
        out["cache_config"] = {
            k: info[k]
            for k in ("num_gpu_blocks", "block_size", "cache_dtype", "gpu_memory_utilization", "enable_prefix_caching")
            if k in info
        }
    return out
