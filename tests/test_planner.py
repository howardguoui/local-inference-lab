import pytest

from inference_lab.models import PRESETS, ModelSpec, get_model
from inference_lab.planner import (
    GIB,
    LLAMACPP_KV_BYTES,
    kv_bytes_per_token,
    kv_dtype_table,
    plan_llamacpp,
    plan_vllm,
)


def test_kv_bytes_per_token_matches_published_figures():
    # Llama 3 8B: 2 x 32 layers x 8 KV heads x 128 dims x 2 bytes = 128 KiB per token
    assert kv_bytes_per_token(PRESETS["llama-3.1-8b"], 2) == 128 * 1024
    # Qwen2.5-7B's 7:1 grouped-query attention: 56 KiB per token
    assert kv_bytes_per_token(PRESETS["qwen2.5-7b"], 2) == 56 * 1024
    assert LLAMACPP_KV_BYTES["q8_0"] == pytest.approx(1.0625)  # 32 int8 values + one fp16 scale


def test_hf_config_loader_handles_missing_head_dim_and_nested_text_config(tmp_path):
    cfg = {
        "text_config": {
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "hidden_size": 3584,
            "model_type": "qwen2",
        }
    }
    m = ModelSpec.from_hf_config(cfg, name="x")
    assert (m.n_layers, m.n_kv_heads, m.head_dim, m.gqa_ratio) == (28, 4, 128, 7)
    (tmp_path / "config.json").write_text('{"num_hidden_layers": 2, "num_attention_heads": 4, "hidden_size": 256}')
    m2 = get_model(str(tmp_path))
    assert m2.n_kv_heads == 4 and m2.head_dim == 64  # no GQA field means plain multi-head attention
    with pytest.raises(KeyError, match="Presets"):
        get_model("no-such-model")


def test_vllm_plan_budget_blocks_and_fp8():
    m = PRESETS["qwen2.5-7b"]
    p = plan_vllm(m, gpu_gib=16, weights_gib=5.2, max_model_len=32768)
    assert p.kv_budget_gib == pytest.approx(16 * 0.9 - 5.2 - 1.5)
    assert p.kv_tokens == p.kv_blocks * 16
    assert p.kv_tokens * p.kv_bytes_per_token <= p.kv_budget_gib * GIB
    assert p.fits and p.max_concurrent_at_max_len == p.kv_tokens // 32768
    fp8 = plan_vllm(m, gpu_gib=16, weights_gib=5.2, max_model_len=32768, kv_cache_dtype="fp8")
    assert abs(fp8.kv_blocks - 2 * p.kv_blocks) <= 1


def test_vllm_plan_reports_when_a_sequence_cannot_fit():
    p = plan_vllm(PRESETS["llama-3.1-8b"], gpu_gib=16, weights_gib=14.5, max_model_len=131072)
    assert not p.fits and "refuse to start" in p.advice and p.kv_budget_gib == 0


def test_llamacpp_partial_offload_uses_the_largest_ngl_that_fits():
    m = PRESETS["qwen2.5-32b"]
    p = plan_llamacpp(m, gguf_gib=18.5, ctx=8192, gpu_gib=16)
    assert not p.full_offload and 0 < p.gpu_layers < p.total_layers
    assert p.vram_estimate_gib <= 16
    one_more = (p.gpu_layers + 1) * (p.weights_per_layer_gib + p.kv_per_layer_gib) + p.reserve_gib
    assert one_more > 16
    q8 = plan_llamacpp(m, gguf_gib=18.5, ctx=8192, gpu_gib=16, cache_type_k="q8_0", cache_type_v="q8_0")
    assert q8.gpu_layers >= p.gpu_layers  # a smaller KV cache never costs GPU layers


def test_llamacpp_full_offload_returns_ngl_99():
    p = plan_llamacpp(PRESETS["qwen2.5-7b"], gguf_gib=4.36, ctx=32768, gpu_gib=16)
    assert p.full_offload and p.gpu_layers == 99 and "fits" in p.advice
    assert p.kv_total_gib == pytest.approx(1.75)


def test_kv_dtype_table_drops_duplicates():
    rows = kv_dtype_table(PRESETS["qwen2.5-7b"], 32768)
    assert {(r["engine"], r["dtype"]) for r in rows} >= {("vllm", "float16"), ("vllm", "fp8"), ("llama.cpp", "q8_0")}
    assert len({(r["engine"], r["kib_per_token"]) for r in rows}) == len(rows)
