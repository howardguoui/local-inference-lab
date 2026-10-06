from inference_lab import prom

VLLM = """# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="Qwen/Qwen2.5-7B-Instruct-AWQ"} 0.42
vllm:kv_cache_usage_perc{engine="1",model_name="Qwen/Qwen2.5-7B-Instruct-AWQ"} 0.61
vllm:num_requests_running{engine="0"} 3.0
vllm:num_requests_running{engine="1"} 2.0
vllm:num_requests_waiting{engine="0"} 1.0
vllm:num_preemptions_total{engine="0"} 7.0
vllm:prefix_cache_hits_total{engine="0"} 300.0
vllm:prefix_cache_queries_total{engine="0"} 1200.0
vllm:cache_config_info{block_size="16",cache_dtype="fp8",gpu_memory_utilization="0.9",num_gpu_blocks="18022"} 1.0
vllm:time_to_first_token_seconds_bucket{le="+Inf",engine="0"} 12.0
vllm:some_gauge NaN
"""

LLAMACPP = """# HELP llamacpp:kv_cache_usage_ratio KV-cache usage. 1 means 100 percent usage.
llamacpp:kv_cache_usage_ratio 0.25
llamacpp:requests_processing 2
llamacpp:requests_deferred 0
"""


def test_parse_labels_values_and_specials():
    samples = prom.parse(VLLM)
    kv = [s for s in samples if s.name == "vllm:kv_cache_usage_perc"]
    assert [s.value for s in kv] == [0.42, 0.61] and kv[0].labels["engine"] == "0"
    inf = next(s for s in samples if s.name.endswith("_bucket"))
    assert inf.labels["le"] == "+Inf"
    assert prom.parse('x{a="say \\"hi\\""} 1')[0].labels["a"] == 'say "hi"'


def test_summarize_vllm():
    s = prom.summarize(prom.parse(VLLM))
    assert s["kv_cache_usage"] == 0.61  # the fuller engine
    assert s["requests_running"] == 5 and s["requests_waiting"] == 1
    assert s["preemptions_total"] == 7 and s["prefix_cache_hit_rate"] == 0.25
    assert s["cache_config"] == {
        "num_gpu_blocks": "18022",
        "block_size": "16",
        "cache_dtype": "fp8",
        "gpu_memory_utilization": "0.9",
    }


def test_summarize_llamacpp_and_old_vllm_name():
    s = prom.summarize(prom.parse(LLAMACPP))
    assert s["kv_cache_usage"] == 0.25 and s["requests_running"] == 2 and s["preemptions_total"] is None
    assert "cache_config" not in s
    old = prom.summarize(prom.parse('vllm:gpu_cache_usage_perc{model_name="m"} 0.9'))
    assert old["kv_cache_usage"] == 0.9
