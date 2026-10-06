"""VRAM planning for one GPU: how much KV cache fits, and how many layers to offload.

KV cache per token = 2 (K and V) x layers x KV heads x head dim x bytes per value.
Everything else here is budgeting around that number:

- vLLM pre-allocates `gpu_memory_utilization x total VRAM`, loads the weights, reserves
  activation and CUDA-graph memory, and turns the rest into fixed 16-token KV blocks
  (PagedAttention). The plan predicts that block count; `inference-lab bench` reads the
  real one from vLLM's `cache_config_info` metric so the two can be compared.
- llama.cpp with `-ngl N` puts the output head and the LAST N-1 transformer blocks (each
  with its KV cache) on the GPU; the first blocks and the token embeddings stay on the CPU.
  The plan finds the largest N that fits, which is how a model bigger than VRAM still
  runs. (Recent llama.cpp can pick N itself with `--fit`; this does it in the open.)

Estimates, not guarantees: activation and compute-buffer sizes vary by batch size and
version, so the overhead terms are parameters with conservative defaults.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .gguf_layout import GgufLayout
from .models import ModelSpec

GIB = 1024**3

# Bytes per stored K or V value.
VLLM_KV_BYTES = {"auto": 2.0, "float16": 2.0, "bfloat16": 2.0, "fp8": 1.0, "fp8_e4m3": 1.0, "fp8_e5m2": 1.0}
# llama.cpp block-quantized types store 32 values plus scale(s) per block.
LLAMACPP_KV_BYTES = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34 / 32,
    "q5_1": 24 / 32,
    "q5_0": 22 / 32,
    "q4_1": 20 / 32,
    "q4_0": 18 / 32,
    "iq4_nl": 18 / 32,
}


def kv_bytes_per_token(model: ModelSpec, k_bytes: float, v_bytes: float | None = None) -> float:
    v_bytes = k_bytes if v_bytes is None else v_bytes
    return model.n_layers * model.n_kv_heads * model.head_dim * (k_bytes + v_bytes)


@dataclass
class VllmPlan:
    model: str
    kv_cache_dtype: str
    kv_bytes_per_token: float
    gpu_gib: float
    gpu_memory_utilization: float
    weights_gib: float
    overhead_gib: float
    kv_budget_gib: float
    kv_tokens: int
    kv_blocks: int
    max_model_len: int
    max_concurrent_at_max_len: int
    fits: bool
    advice: str

    def as_dict(self) -> dict:
        return asdict(self)


def plan_vllm(
    model: ModelSpec,
    gpu_gib: float,
    weights_gib: float,
    max_model_len: int,
    gpu_memory_utilization: float = 0.90,
    kv_cache_dtype: str = "auto",
    overhead_gib: float = 1.5,
    block_size: int = 16,
) -> VllmPlan:
    """Predict vLLM's KV cache allocation. overhead_gib covers activations at the default
    batch size plus CUDA graphs; check it against the startup log's KV cache line."""
    per_token = kv_bytes_per_token(model, VLLM_KV_BYTES[kv_cache_dtype])
    budget = gpu_gib * gpu_memory_utilization - weights_gib - overhead_gib
    blocks = max(0, int(budget * GIB // (per_token * block_size)))
    tokens = blocks * block_size
    fits = tokens >= max_model_len
    concurrent = tokens // max_model_len if max_model_len else 0
    if not fits:
        advice = (
            f"vLLM will refuse to start: one {max_model_len}-token sequence needs "
            f"{max_model_len * per_token / GIB:.2f} GiB of KV cache but only {max(budget, 0):.2f} GiB is left. "
            "Lower --max-model-len, use --kv-cache-dtype fp8, or raise --gpu-memory-utilization."
        )
    elif kv_cache_dtype in ("auto", "float16", "bfloat16"):
        advice = (
            f"--kv-cache-dtype fp8 would hold about {2 * tokens:,} tokens ({2 * concurrent} full-length sequences)."
        )
    else:
        advice = "FP8 KV cache halves memory per token; check answer quality on your eval set before relying on it."
    return VllmPlan(
        model=model.name,
        kv_cache_dtype=kv_cache_dtype,
        kv_bytes_per_token=per_token,
        gpu_gib=gpu_gib,
        gpu_memory_utilization=gpu_memory_utilization,
        weights_gib=round(weights_gib, 2),
        overhead_gib=overhead_gib,
        kv_budget_gib=round(max(budget, 0.0), 2),
        kv_tokens=tokens,
        kv_blocks=blocks,
        max_model_len=max_model_len,
        max_concurrent_at_max_len=concurrent,
        fits=fits,
        advice=advice,
    )


@dataclass
class OffloadPlan:
    model: str
    layout_source: str  # "gguf" (tensor table read) or "estimate" (from file size and model shape)
    ctx: int
    cache_type_k: str
    cache_type_v: str
    gpu_gib: float
    reserve_gib: float
    ngl: int  # value for -ngl (99 = everything)
    blocks_on_gpu: int
    total_blocks: int
    full_offload: bool
    vram_estimate_gib: float
    cpu_weights_gib: float  # stays in system RAM: token embeddings plus blocks not offloaded
    kv_per_block_gib: float
    advice: str

    def as_dict(self) -> dict:
        return asdict(self)


def _gpu_bytes(layout: GgufLayout, ngl: int, kv_block: float) -> float:
    """VRAM for -ngl ngl, excluding the reserve. llama.cpp (llama-model.cpp) assigns the
    output head the first GPU slot, then the last ngl-1 blocks; token_embd stays on the CPU."""
    if ngl <= 0:
        return 0.0
    blocks = min(ngl - 1, layout.n_layers)
    on_gpu = layout.block_bytes[layout.n_layers - blocks :] if blocks else []
    return layout.output_bytes + sum(on_gpu) + blocks * kv_block


def plan_llamacpp(
    model: ModelSpec,
    layout: GgufLayout,
    ctx: int,
    gpu_gib: float,
    cache_type_k: str = "f16",
    cache_type_v: str = "f16",
    reserve_gib: float = 1.0,
) -> OffloadPlan:
    """Largest -ngl that fits in gpu_gib. reserve_gib covers llama.cpp's compute buffer and
    anything else on the GPU (a desktop's display). ctx is the total -c shared by all slots."""
    kv_block = (
        ctx * model.n_kv_heads * model.head_dim * (LLAMACPP_KV_BYTES[cache_type_k] + LLAMACPP_KV_BYTES[cache_type_v])
    )
    budget = (gpu_gib - reserve_gib) * GIB
    units = layout.n_layers + 1
    ngl = 0
    for n in range(1, units + 1):
        if _gpu_bytes(layout, n, kv_block) > budget:
            break
        ngl = n
    full = ngl >= units
    blocks = max(0, ngl - 1)
    vram = _gpu_bytes(layout, ngl, kv_block) / GIB + reserve_gib
    cpu = layout.token_embd_bytes + sum(layout.block_bytes[: layout.n_layers - blocks]) + layout.other_bytes
    if full:
        advice = f"Everything fits (-ngl 99) with about {gpu_gib - vram:.1f} GiB to spare for a longer context."
    elif ngl == 0:
        advice = "Even the output head doesn't fit; use a smaller quantization or a shorter context."
    else:
        advice = (
            f"-ngl {ngl}: the output head and the last {blocks} of {layout.n_layers} blocks on the GPU, "
            f"{cpu / GIB:.1f} GiB of weights in system RAM. Generation speed is then bound by RAM bandwidth."
        )
        if LLAMACPP_KV_BYTES[cache_type_k] + LLAMACPP_KV_BYTES[cache_type_v] > 2 * LLAMACPP_KV_BYTES["q8_0"]:
            advice += " A quantized KV cache (-ctk q8_0 -ctv q8_0, with -fa on) frees room for more blocks."
    return OffloadPlan(
        model=model.name,
        layout_source=layout.source,
        ctx=ctx,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        gpu_gib=round(gpu_gib, 2),
        reserve_gib=reserve_gib,
        ngl=99 if full else ngl,
        blocks_on_gpu=blocks,
        total_blocks=layout.n_layers,
        full_offload=full,
        vram_estimate_gib=round(vram, 2),
        cpu_weights_gib=round(cpu / GIB, 2),
        kv_per_block_gib=round(kv_block / GIB, 4),
        advice=advice,
    )


def kv_dtype_table(model: ModelSpec, tokens: int) -> list[dict]:
    """KV cache size for one sequence of `tokens` under each storage type."""
    rows = []
    for engine, table in (("vllm", VLLM_KV_BYTES), ("llama.cpp", LLAMACPP_KV_BYTES)):
        seen = set()
        for dtype, b in table.items():
            if (engine, b) in seen or dtype in ("auto", "fp8_e4m3", "bf16"):
                continue
            seen.add((engine, b))
            per_tok = kv_bytes_per_token(model, b)
            rows.append(
                {
                    "engine": engine,
                    "dtype": dtype,
                    "kib_per_token": round(per_tok / 1024, 1),
                    "gib": round(per_tok * tokens / GIB, 3),
                }
            )
    return rows
