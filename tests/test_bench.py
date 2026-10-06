import asyncio

import httpx

from inference_lab.backends import Backend, health
from inference_lab.bench import make_prompts, percentile, run_benchmark, stream_chat
from inference_lab.gpu import GpuSampler, gpu_snapshot


def test_prompts_are_unique_and_sized():
    prompts = make_prompts(3, 512, offset=40)
    assert len(set(prompts)) == 3 and prompts[0].startswith("Request 40.")
    assert 350 < len(prompts[0].split()) < 450  # ~0.75 words per token


def test_percentile_nearest_rank():
    vals = [5, 1, 4, 2, 3]
    assert percentile(vals, 50) == 3 and percentile(vals, 95) == 5 and percentile(vals, 0) == 1
    assert percentile([], 50) is None and percentile([None, 2.0], 50) == 2.0


def test_stream_chat_measures_ttft_and_tokens(server):
    async def go():
        async with httpx.AsyncClient(base_url=server.url + "/v1") as c:
            ok = await stream_chat(c, "fake/qwen-7b", "hi", 10)
            bad = await stream_chat(c, "wrong-model", "hi", 10)
        return ok, bad

    ok, bad = asyncio.run(go())
    assert ok.ok and ok.output_tokens == 10 and ok.tokens_from_usage and ok.prompt_tokens == 600
    assert ok.finish_reason == "length"  # ran to max_tokens because EOS was ignored
    assert 0.02 < ok.ttft_s < ok.e2e_s and ok.tpot_s > 0
    assert not bad.ok and bad.error.startswith("HTTP 404")


def test_errors_inside_a_200_stream_and_early_eos(server):
    async def go():
        async with httpx.AsyncClient(base_url=server.url + "/v1") as c:
            err = await stream_chat(c, "fake/qwen-7b", "FAIL please", 10)
            eos = await stream_chat(c, "fake/qwen-7b", "hi", 10, ignore_eos=False)
        return err, eos

    err, eos = asyncio.run(go())
    assert not err.ok and "context length exceeded" in err.error
    assert eos.ok and eos.output_tokens == 5 and eos.finish_reason == "stop"


def test_benchmark_levels_metrics_and_gpu(server, nvml):
    backend = Backend("fake", "vllm", server.url + "/v1", server.url + "/metrics")
    result = asyncio.run(
        run_benchmark(
            backend,
            [1, 8],
            requests_per_level=8,
            prompt_tokens=64,
            max_tokens=20,
            warmup=1,
            nvml=nvml,
            metrics_interval=0.01,
        )
    ).as_dict()
    assert result["model"] == "fake/qwen-7b"
    assert result["gpu"]["name"] == "NVIDIA GeForce RTX 5070 Ti"
    assert result["server"]["cache_config"]["num_gpu_blocks"] == "9011"
    one, eight = result["levels"]
    for lv in (one, eight):
        assert lv["errors"] == 0 and lv["requests"] == 8 and lv["output_tokens"] == 160 and lv["tokens_from_usage"]
        assert lv["ttft_p50_ms"] <= lv["ttft_p95_ms"] and lv["peak_vram_gib"] and lv["peak_power_w"] == 250
        assert lv["mean_output_tokens"] == 20 and lv["mean_prompt_tokens"] == 600
        assert lv["finish_reasons"] == {"length": 8}
    assert one["max_in_flight"] == 1 and eight["max_in_flight"] == 8
    assert eight["throughput_tok_s"] > 2 * one["throughput_tok_s"]  # batching pays off
    assert eight["peak_kv_cache_usage"] > one["peak_kv_cache_usage"]
    assert one["preemptions"] == 0 and eight["preemptions"] > 0  # 8 > the fake cache's 4 slots


def test_unreachable_server_is_reported_not_raised():
    backend = Backend("down", "vllm", "http://127.0.0.1:9/v1")
    h = asyncio.run(health(backend))
    assert h["up"] is False and "ConnectError" in h["error"]


def test_gpu_sampler_and_snapshot_without_nvidia_driver():
    from tests.conftest import FakeNvml

    assert gpu_snapshot(nvml=FakeNvml(fail_init=True)) is None
    stats = GpuSampler(nvml=FakeNvml(fail_init=True)).start().stop()
    assert stats.samples == 0 and stats.peak_memory_gib is None


def test_gpu_sampler_tracks_peak(nvml):
    import time

    s = GpuSampler(interval=0.01, nvml=nvml).start()
    time.sleep(0.15)
    stats = s.stop()
    assert stats.samples >= 5 and stats.peak_memory_gib == 12.0 and stats.mean_utilization_pct == 90
    snap = gpu_snapshot(nvml=nvml)
    assert snap["memory_total_gib"] == 16 and snap["power_w"] == 250 and snap["temperature_c"] == 64


def test_refuses_levels_that_could_never_reach_their_concurrency(server):
    import pytest

    backend = Backend("fake", "vllm", server.url + "/v1", server.url + "/metrics")
    with pytest.raises(ValueError, match="below the top concurrency"):
        asyncio.run(run_benchmark(backend, [1, 48], requests_per_level=32))
