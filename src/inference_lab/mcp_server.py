"""MCP server: lets an agent (Claude Code, Claude Desktop, any MCP client) inspect the GPU,
size a deployment before launching it, check what the inference servers are doing,
and run a benchmark, all as tool calls.

    inference-lab mcp            # stdio, for Claude Code / Claude Desktop
    inference-lab mcp --http     # streamable HTTP on http://127.0.0.1:8765/mcp
"""

from __future__ import annotations

from fastmcp import FastMCP

from . import prom, report
from .backends import health, load_backends
from .bench import run_benchmark
from .gpu import gpu_snapshot
from .models import PRESETS, get_model
from .planner import VLLM_KV_BYTES, kv_dtype_table, plan_llamacpp, plan_vllm

mcp = FastMCP(
    "local-inference-lab",
    instructions=(
        "Tools for running LLMs on a local NVIDIA GPU. Check gpu_status first; use plan_vllm or plan_llamacpp "
        "before starting a server to see what fits in VRAM; use backend_metrics to watch KV cache pressure; "
        "run_benchmark measures throughput and latency and saves the result."
    ),
)


@mcp.tool
def gpu_status() -> dict:
    """Current GPU name, VRAM used and free, utilization, temperature and power."""
    return gpu_snapshot() or {"error": "No NVIDIA GPU or driver found (NVML unavailable)."}


@mcp.tool
def list_models() -> list[dict]:
    """Model presets the planners know (attention shape only; any config.json path also works)."""
    return [
        {"name": m.name, "layers": m.n_layers, "kv_heads": m.n_kv_heads, "head_dim": m.head_dim, "params_b": m.params_b}
        for m in PRESETS.values()
    ]


@mcp.tool
def kv_cache_size(model: str, tokens: int) -> list[dict]:
    """KV cache size of one sequence of `tokens` tokens for each storage type (vLLM and llama.cpp)."""
    return kv_dtype_table(get_model(model), tokens)


@mcp.tool
def plan_vllm_deployment(
    model: str,
    weights_gib: float,
    max_model_len: int = 8192,
    gpu_gib: float = 16.0,
    gpu_memory_utilization: float = 0.90,
    kv_cache_dtype: str = "auto",
    overhead_gib: float = 1.5,
) -> dict:
    """Predict how many tokens of KV cache vLLM will allocate and how many full-length sequences fit.
    weights_gib: size of the checkpoint on disk. kv_cache_dtype: one of auto, float16, bfloat16, fp8."""
    if kv_cache_dtype not in VLLM_KV_BYTES:
        raise ValueError(f"kv_cache_dtype must be one of {sorted(VLLM_KV_BYTES)}")
    return plan_vllm(
        get_model(model), gpu_gib, weights_gib, max_model_len, gpu_memory_utilization, kv_cache_dtype, overhead_gib
    ).as_dict()


@mcp.tool
def plan_llamacpp_offload(
    model: str,
    gguf_gib: float,
    ctx: int = 8192,
    gpu_gib: float = 16.0,
    cache_type_k: str = "f16",
    cache_type_v: str = "f16",
    reserve_gib: float = 1.0,
) -> dict:
    """Largest -ngl (layers on GPU) for a GGUF model, with the VRAM it will use. Works for models
    bigger than VRAM: the remaining layers run on the CPU."""
    return plan_llamacpp(get_model(model), gguf_gib, ctx, gpu_gib, cache_type_k, cache_type_v, reserve_gib).as_dict()


@mcp.tool
async def list_backends() -> list[dict]:
    """Configured inference servers, whether each is up, and the model it serves."""
    return [await health(b) for b in load_backends().values()]


@mcp.tool
async def backend_metrics(backend: str) -> dict:
    """Live KV cache fill, running/waiting requests, preemptions and cache allocation from a server's /metrics."""
    import httpx

    b = load_backends()[backend]
    if not b.metrics_url:
        return {"error": f"{backend} exposes no Prometheus metrics"}
    async with httpx.AsyncClient() as client:
        resp = await client.get(b.metrics_url, timeout=5)
        resp.raise_for_status()
    return prom.summarize(prom.parse(resp.text))


@mcp.tool(name="run_benchmark")
async def run_benchmark_tool(
    backend: str,
    concurrency: list[int] | None = None,
    requests_per_level: int = 8,
    prompt_tokens: int = 512,
    max_tokens: int = 128,
    label: str | None = None,
) -> dict:
    """Benchmark one server at each concurrency level; saves the run and returns the per-level summary."""
    result = await run_benchmark(
        load_backends()[backend],
        concurrency or [1, 4],
        requests_per_level=requests_per_level,
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        label=label,
    )
    data = result.as_dict()
    path = report.save(data)
    report.write_markdown()
    return {"saved": str(path), "model": data["model"], "levels": data["levels"]}


@mcp.resource("lab://results/latest")
def latest_results() -> str:
    """The comparison table across all saved benchmark runs (Markdown)."""
    return report.markdown(report.load_all())


def main(http: bool = False, port: int = 8765) -> None:
    if http:
        mcp.run(transport="http", host="127.0.0.1", port=port)
    else:
        mcp.run(show_banner=False)
