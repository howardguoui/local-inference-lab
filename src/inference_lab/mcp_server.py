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
from .gguf_layout import estimate_layout, read_gguf
from .gpu import default_gpu_gib, gpu_snapshot
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
    gpu_gib: float | None = None,
    gpu_memory_utilization: float = 0.90,
    kv_cache_dtype: str = "auto",
    overhead_gib: float = 1.5,
) -> dict:
    """Predict how many tokens of KV cache vLLM will allocate and how many full-length sequences fit.
    weights_gib: size of the checkpoint on disk. kv_cache_dtype: one of auto, float16, bfloat16, fp8.
    gpu_gib defaults to the GPU's total memory (NVML)."""
    if kv_cache_dtype not in VLLM_KV_BYTES:
        raise ValueError(f"kv_cache_dtype must be one of {sorted(VLLM_KV_BYTES)}")
    gpu = gpu_gib if gpu_gib is not None else default_gpu_gib("total")[0]
    return plan_vllm(
        get_model(model), gpu, weights_gib, max_model_len, gpu_memory_utilization, kv_cache_dtype, overhead_gib
    ).as_dict()


@mcp.tool
def plan_llamacpp_offload(
    gguf_path: str | None = None,
    model: str | None = None,
    gguf_gib: float | None = None,
    ctx: int = 8192,
    gpu_gib: float | None = None,
    cache_type_k: str = "f16",
    cache_type_v: str = "f16",
    reserve_gib: float = 1.0,
) -> dict:
    """Largest -ngl (output head + last N-1 blocks on GPU) for a GGUF model, with its VRAM estimate.
    Works for models bigger than VRAM: the remaining blocks run on the CPU. Give gguf_path for exact
    tensor sizes, or model + gguf_gib for an estimate. gpu_gib defaults to free VRAM (NVML)."""
    if gguf_path:
        layout = read_gguf(gguf_path)
        spec = get_model(model) if model else layout.spec
    elif model and gguf_gib:
        spec = get_model(model)
        layout = estimate_layout(spec, int(gguf_gib * 1024**3))
    else:
        raise ValueError("pass gguf_path, or model and gguf_gib")
    gpu = gpu_gib if gpu_gib is not None else default_gpu_gib("free")[0]
    return plan_llamacpp(spec, layout, ctx, gpu, cache_type_k, cache_type_v, reserve_gib).as_dict()


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
