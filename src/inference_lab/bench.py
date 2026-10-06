"""Load generator for OpenAI-compatible servers (vLLM, llama.cpp, Ollama).

For each concurrency level it streams `requests` chat completions through that many
parallel workers and measures what users feel and what the GPU pays:

- time to first token (TTFT): queueing plus prefill
- time per output token (TPOT): decode speed once a stream has started
- aggregate throughput: output tokens per second across all streams
- peak VRAM, GPU utilization and power (NVML), and, from /metrics, peak KV cache
  fill and preemptions (vLLM evicting sequences because the cache ran out)

Every prompt starts with a unique request number so prefix caching can't make a
repeated prompt look free, and generation ignores end-of-sequence by default so every
request produces exactly max_tokens tokens on every server (different quantizations
would otherwise stop at different lengths and make latencies incomparable).
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

import httpx

from . import prom
from .backends import Backend, discover_model
from .gpu import GpuSampler, gpu_snapshot

_PASSAGE = (
    "A GPU serves a language model in two phases. Prefill reads the whole prompt at once and is limited by "
    "compute. Decode then produces one token per step for every active sequence and is limited by how fast "
    "weights and the key-value cache can be read from memory. Batching many sequences together amortizes the "
    "weight reads, which is why throughput climbs with concurrency until the KV cache runs out of room. "
)


def make_prompts(n: int, prompt_tokens: int, offset: int = 0) -> list[str]:
    """n distinct prompts of roughly prompt_tokens tokens (about 0.75 words per token)."""
    words = _PASSAGE.split()
    target = max(8, int(prompt_tokens * 0.75))
    body = " ".join(words[i % len(words)] for i in range(target))
    return [
        f"Request {offset + i}. Read the passage, then explain in your own words why decode is memory-bound.\n\n{body}"
        for i in range(n)
    ]


@dataclass
class RequestResult:
    ok: bool
    e2e_s: float
    ttft_s: float | None = None
    output_tokens: int = 0
    prompt_tokens: int | None = None
    tokens_from_usage: bool = False
    finish_reason: str | None = None
    error: str | None = None

    @property
    def tpot_s(self) -> float | None:
        if self.ttft_s is None or self.output_tokens < 2:
            return None
        return (self.e2e_s - self.ttft_s) / (self.output_tokens - 1)


async def stream_chat(
    client: httpx.AsyncClient, model: str, prompt: str, max_tokens: int, ignore_eos: bool = True
) -> RequestResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        payload["ignore_eos"] = True  # vLLM and llama.cpp extension; Ollama ignores unknown fields
    t0 = time.perf_counter()
    first: float | None = None
    chunks = 0
    usage: dict | None = None
    finish: str | None = None
    try:
        async with client.stream("POST", "/chat/completions", json=payload, timeout=300) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")[:200]
                return RequestResult(False, time.perf_counter() - t0, error=f"HTTP {resp.status_code}: {body}")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if obj.get("error") or obj.get("object") == "error":  # servers report failures mid-stream
                    err = obj.get("error", obj)
                    msg = err.get("message", err) if isinstance(err, dict) else err
                    return RequestResult(False, time.perf_counter() - t0, error=f"stream error: {msg}"[:200])
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices") or []:
                    finish = choice.get("finish_reason") or finish
                    delta = choice.get("delta") or {}
                    # reasoning models stream their thinking in a separate field; it's still decode work
                    if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                        if first is None:
                            first = time.perf_counter()
                        chunks += 1
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        return RequestResult(False, time.perf_counter() - t0, error=f"{type(exc).__name__}: {exc}"[:200])
    end = time.perf_counter()
    from_usage = bool(usage and usage.get("completion_tokens"))
    tokens = usage["completion_tokens"] if from_usage else chunks
    if tokens == 0 or first is None:
        return RequestResult(False, end - t0, error="stream ended without generating any tokens")
    return RequestResult(
        ok=True,
        e2e_s=end - t0,
        ttft_s=first - t0,
        output_tokens=tokens,
        prompt_tokens=(usage or {}).get("prompt_tokens"),
        tokens_from_usage=from_usage,
        finish_reason=finish,
    )


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile; q in [0, 100]."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    k = max(0, min(len(vals) - 1, math.ceil(q / 100 * len(vals)) - 1))
    return vals[k]


def _r(x: float | None, scale: float = 1.0, nd: int = 1) -> float | None:
    return None if x is None else round(x * scale, nd)


@dataclass
class LevelResult:
    concurrency: int
    requests: int
    errors: int
    wall_s: float
    throughput_tok_s: float
    requests_per_s: float
    ttft_p50_ms: float | None
    ttft_p95_ms: float | None
    tpot_p50_ms: float | None
    tpot_p95_ms: float | None
    e2e_p50_s: float | None
    e2e_p95_s: float | None
    output_tokens: int
    tokens_from_usage: bool
    mean_output_tokens: float | None = None
    mean_prompt_tokens: float | None = None  # as the server counted them, chat template included
    finish_reasons: dict[str, int] = field(default_factory=dict)
    peak_vram_gib: float | None = None
    mean_gpu_util_pct: float | None = None
    peak_power_w: float | None = None
    peak_kv_cache_usage: float | None = None
    preemptions: float | None = None
    first_error: str | None = None


class MetricsPoller:
    """Polls /metrics during a level; tracks peak KV cache fill and the preemption counter's change."""

    def __init__(self, client: httpx.AsyncClient, url: str | None, interval: float = 0.5):
        self.client, self.url, self.interval = client, url, interval
        self.peak_kv: float | None = None
        self.preempt_start: float | None = None
        self.preempt_end: float | None = None
        self.last: dict = {}
        self._task: asyncio.Task | None = None

    async def read(self) -> dict:
        if not self.url:
            return {}
        try:
            resp = await self.client.get(self.url, timeout=5)
            resp.raise_for_status()
        except httpx.HTTPError:
            return {}
        summary = prom.summarize(prom.parse(resp.text))
        kv = summary.get("kv_cache_usage")
        if kv is not None:
            self.peak_kv = kv if self.peak_kv is None else max(self.peak_kv, kv)
        self.last = summary
        return summary

    async def _loop(self) -> None:
        while True:
            await self.read()
            await asyncio.sleep(self.interval)

    async def __aenter__(self) -> MetricsPoller:
        self.preempt_start = (await self.read()).get("preemptions_total")
        if self.url:
            self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.preempt_end = (await self.read()).get("preemptions_total")

    @property
    def preemptions(self) -> float | None:
        if self.preempt_start is None or self.preempt_end is None:
            return None
        return self.preempt_end - self.preempt_start


async def run_level(
    client: httpx.AsyncClient,
    model: str,
    concurrency: int,
    n_requests: int,
    prompt_tokens: int,
    max_tokens: int,
    metrics: MetricsPoller,
    sampler: GpuSampler,
    offset: int = 0,
    ignore_eos: bool = True,
) -> LevelResult:
    queue: asyncio.Queue[str] = asyncio.Queue()
    for p in make_prompts(n_requests, prompt_tokens, offset):
        queue.put_nowait(p)
    results: list[RequestResult] = []

    async def worker() -> None:
        while True:
            try:
                prompt = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            results.append(await stream_chat(client, model, prompt, max_tokens, ignore_eos))

    sampler.start()
    async with metrics:
        t0 = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        wall = time.perf_counter() - t0
    gpu = sampler.stop()

    ok = [r for r in results if r.ok]
    out_tokens = sum(r.output_tokens for r in ok)
    errors = [r for r in results if not r.ok]
    prompt_counts = [r.prompt_tokens for r in ok if r.prompt_tokens]
    finishes: dict[str, int] = {}
    for r in ok:
        finishes[r.finish_reason or "unknown"] = finishes.get(r.finish_reason or "unknown", 0) + 1
    return LevelResult(
        concurrency=concurrency,
        requests=len(results),
        errors=len(errors),
        wall_s=round(wall, 2),
        throughput_tok_s=round(out_tokens / wall, 1) if wall else 0.0,
        requests_per_s=round(len(ok) / wall, 2) if wall else 0.0,
        ttft_p50_ms=_r(percentile([r.ttft_s for r in ok], 50), 1000),
        ttft_p95_ms=_r(percentile([r.ttft_s for r in ok], 95), 1000),
        tpot_p50_ms=_r(percentile([r.tpot_s for r in ok], 50), 1000, 2),
        tpot_p95_ms=_r(percentile([r.tpot_s for r in ok], 95), 1000, 2),
        e2e_p50_s=_r(percentile([r.e2e_s for r in ok], 50), 1, 2),
        e2e_p95_s=_r(percentile([r.e2e_s for r in ok], 95), 1, 2),
        output_tokens=out_tokens,
        tokens_from_usage=bool(ok) and all(r.tokens_from_usage for r in ok),
        mean_output_tokens=round(out_tokens / len(ok), 1) if ok else None,
        mean_prompt_tokens=round(sum(prompt_counts) / len(prompt_counts), 1) if prompt_counts else None,
        finish_reasons=finishes,
        peak_vram_gib=gpu.peak_memory_gib,
        mean_gpu_util_pct=gpu.mean_utilization_pct,
        peak_power_w=gpu.peak_power_w,
        peak_kv_cache_usage=_r(metrics.peak_kv, 1, 3),
        preemptions=metrics.preemptions,
        first_error=errors[0].error if errors else None,
    )


@dataclass
class BenchResult:
    label: str
    backend: dict
    model: str
    started_at: str
    prompt_tokens: int
    max_tokens: int
    ignore_eos: bool
    gpu: dict | None
    server: dict
    levels: list[LevelResult] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


async def run_benchmark(
    backend: Backend,
    concurrency: list[int],
    requests_per_level: int = 32,
    prompt_tokens: int = 512,
    max_tokens: int = 256,
    label: str | None = None,
    warmup: int = 2,
    transport: httpx.AsyncBaseTransport | None = None,
    sampler_factory=GpuSampler,
    nvml=None,
    metrics_interval: float = 0.5,
    ignore_eos: bool = True,
) -> BenchResult:
    async with httpx.AsyncClient(base_url=backend.base_url, transport=transport) as client:
        model = backend.model or await discover_model(client)
        for p in make_prompts(warmup, prompt_tokens, offset=10_000):  # load weights, build CUDA graphs
            await stream_chat(client, model, p, 16)
        server = await MetricsPoller(client, backend.metrics_url).read()
        result = BenchResult(
            label=label or backend.name,
            backend=backend.as_dict(),
            model=model,
            started_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            ignore_eos=ignore_eos,
            gpu=gpu_snapshot(nvml=nvml),
            server=server,
        )
        offset = 0
        for c in concurrency:
            level = await run_level(
                client,
                model,
                c,
                requests_per_level,
                prompt_tokens,
                max_tokens,
                MetricsPoller(client, backend.metrics_url, metrics_interval),
                sampler_factory(nvml=nvml) if nvml is not None else sampler_factory(),
                offset,
                ignore_eos,
            )
            result.levels.append(level)
            offset += requests_per_level
        return result
