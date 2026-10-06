"""VRAM planning for one GPU: how much KV cache fits, and how many layers to offload.

KV cache per token = 2 (K and V) x layers x KV heads x head dim x bytes per value.
Everything else here is budgeting around that number:

- vLLM pre-allocates `gpu_memory_utilization x total VRAM`, loads the weights, reserves
  activation and CUDA-graph memory, and turns the rest into fixed 16-token KV blocks
  (PagedAttention). The plan predicts that block count; `inference-lab bench` reads the
  real one from vLLM's `cache_config_info` metric so the two can be compared.
- llama.cpp puts the first `-ngl` layers (weights and their KV cache) on the GPU and the
  rest on the CPU. The plan finds the largest `-ngl` that fits, which is how a model
  bigger than VRAM still runs.

Estimates, not guarantees: activation and compute-buffer sizes vary by batch size and
version, so the overhead terms are parameters with conservative defaults.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

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
# Rough effective bits per weight, for when the file size isn't known yet.
BITS_PER_WEIGHT = {"f16": 16.0, "bf16": 16.0, "fp8": 8.0, "q8_0": 8.5, "q6_k": 6.56, "q5_k_m": 5.69, "q4_k_m": 4.89}


def kv_bytes_per_token(model: ModelSpec, k_bytes: float, v_bytes: float | None = None) -> float:
    v_bytes = k_bytes if v_bytes is None else v_bytes
    return model.n_layers * model.n_kv_heads * model.head_dim * (k_bytes + v_bytes)


def estimate_weights_gib(model: ModelSpec, quant: str) -> float:
    """Rough weight size. Prefer the real file size: AWQ/GPTQ checkpoints keep embeddings
    in 16-bit, so they are larger than params x 4 bits suggests."""
    return model.params_b * 1e9 * BITS_PER_WEIGHT[quant.lower()] / 8 / GIB


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
    gguf_gib: float
    ctx: int
    cache_type_k: str
    cache_type_v: str
    gpu_gib: float
    reserve_gib: float
    weights_per_layer_gib: float
    kv_per_layer_gib: float
    gpu_layers: int  # value for -ngl
    total_layers: int
    full_offload: bool
    vram_estimate_gib: float
    kv_total_gib: float
    advice: str

    def as_dict(self) -> dict:
        return asdict(self)


def plan_llamacpp(
    model: ModelSpec,
    gguf_gib: float,
    ctx: int,
    gpu_gib: float,
    cache_type_k: str = "f16",
    cache_type_v: str = "f16",
    reserve_gib: float = 1.0,
) -> OffloadPlan:
    """Largest -ngl that fits. The GGUF is split evenly over the repeating layers plus one
    extra unit for the embeddings and output head, which llama.cpp offloads last.
    reserve_gib covers the compute buffer and anything else on the GPU (a desktop's display)."""
    kv_token_layer = (
        model.n_kv_heads * model.head_dim * (LLAMACPP_KV_BYTES[cache_type_k] + LLAMACPP_KV_BYTES[cache_type_v])
    )
    kv_layer_gib = ctx * kv_token_layer / GIB
    units = model.n_layers + 1
    w_layer_gib = gguf_gib / units
    available = gpu_gib - reserve_gib
    fit = int(available // (w_layer_gib + kv_layer_gib)) if available > 0 else 0
    gpu_layers = min(fit, units)
    full = gpu_layers >= units
    kv_on_gpu = min(gpu_layers, model.n_layers) * kv_layer_gib
    vram = gpu_layers * w_layer_gib + kv_on_gpu + reserve_gib
    if full:
        spare = gpu_gib - vram
        advice = f"Everything fits on the GPU (-ngl 99) with about {spare:.1f} GiB to spare for a longer context."
    elif gpu_layers == 0:
        advice = "Nothing fits on the GPU; use a smaller quantization or a shorter context."
    else:
        advice = (
            f"Offload {gpu_layers} of {units} layers (-ngl {gpu_layers}); the rest run on the CPU, "
            "so generation speed is bound by system RAM bandwidth. A quantized KV cache "
            "(-ctk q8_0 -ctv q8_0, needs -fa on) frees room for more GPU layers."
        )
    return OffloadPlan(
        model=model.name,
        gguf_gib=gguf_gib,
        ctx=ctx,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        gpu_gib=gpu_gib,
        reserve_gib=reserve_gib,
        weights_per_layer_gib=round(w_layer_gib, 4),
        kv_per_layer_gib=round(kv_layer_gib, 4),
        gpu_layers=99 if full else gpu_layers,
        total_layers=units,
        full_offload=full,
        vram_estimate_gib=round(vram, 2),
        kv_total_gib=round(model.n_layers * kv_layer_gib, 2),
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
