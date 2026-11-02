"""Read the Prometheus /metrics endpoint that vLLM, SGLang (--enable-metrics) and llama.cpp (--metrics) expose.

The numbers that matter for capacity are the KV cache fill level and, for vLLM,
preemptions: when the cache is full vLLM evicts running sequences and recomputes
them later (SGLang calls this retraction), which shows up as latency spikes. Both are read here, along with the
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
# SGLang names are from python/sglang/srt/observability/metrics_collector.py.
KV_USAGE = (
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
    "sglang:token_usage",
    "llamacpp:kv_cache_usage_ratio",
)
RUNNING = ("vllm:num_requests_running", "sglang:num_running_reqs", "llamacpp:requests_processing")
WAITING = ("vllm:num_requests_waiting", "sglang:num_queue_reqs", "llamacpp:requests_deferred")
PREEMPTIONS = ("vllm:num_preemptions_total", "vllm:num_preemptions", "sglang:num_retracted_requests_total")
# Prompt tokens served from the prefix cache, and prompt tokens seen. vLLM counts both when it schedules a
# request; SGLang when a request finishes (cached_tokens_total is split by cache_source, prompt_tokens_total by
# is_streaming; the series are summed). Only deltas over a whole level are compared, so either works.
PREFIX_HITS = ("vllm:prefix_cache_hits_total", "sglang:cached_tokens_total")
PREFIX_QUERIES = ("vllm:prefix_cache_queries_total", "sglang:prompt_tokens_total")


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
    hits = _values(samples, PREFIX_HITS)
    queries = _values(samples, PREFIX_QUERIES)
    if queries and not hits:
        hits = [0.0]  # SGLang exports cached_tokens_total only once some prompt token has hit the cache
    out: dict = {
        "kv_cache_usage": max(kv) if kv else None,  # max across engines
        "requests_running": sum(running) if running else None,
        "requests_waiting": sum(waiting) if waiting else None,
        "preemptions_total": sum(preempt) if preempt else None,
        # counters in prompt tokens since the server started (PREFIX_HITS / PREFIX_QUERIES above)
        "prefix_cache_hits_total": sum(hits) if hits else None,
        "prefix_cache_queries_total": sum(queries) if queries else None,
        "prefix_cache_hit_rate": (sum(hits) / sum(queries)) if hits and queries and sum(queries) else None,
    }
    info = next((s.labels for s in samples if s.name == "vllm:cache_config_info"), None)
    if info:
        out["cache_config"] = {
            k: info[k]
            for k in ("num_gpu_blocks", "block_size", "cache_dtype", "gpu_memory_utilization", "enable_prefix_caching")
            if k in info
        }
    sglang_tokens = _values(samples, ("sglang:max_total_num_tokens",))
    if sglang_tokens:  # SGLang's KV pool size in tokens, the counterpart of vLLM's num_gpu_blocks x block_size
        out["cache_config"] = {"max_total_num_tokens": str(int(max(sglang_tokens)))}
    return out
