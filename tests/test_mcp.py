"""Drive the MCP server the way an agent would, through an in-memory MCP client."""

import asyncio

import pytest
from fastmcp import Client

from inference_lab import gpu
from inference_lab.mcp_server import mcp


async def _call(name: str, args: dict | None = None):
    async with Client(mcp) as c:
        return await c.call_tool(name, args or {}, raise_on_error=False)


def test_tools_are_listed_with_descriptions():
    async def go():
        async with Client(mcp) as c:
            return {t.name: t.description for t in await c.list_tools()}

    tools = asyncio.run(go())
    assert set(tools) == {
        "gpu_status",
        "list_models",
        "kv_cache_size",
        "plan_vllm_deployment",
        "plan_llamacpp_offload",
        "list_backends",
        "backend_metrics",
        "run_benchmark",
        "compare_configs",
    }
    assert all(tools.values())


def test_planning_tools():
    plan = asyncio.run(
        _call(
            "plan_vllm_deployment",
            {"model": "qwen2.5-7b", "weights_gib": 5.2, "max_model_len": 32768, "kv_cache_dtype": "fp8"},
        )
    ).data
    assert plan["fits"] and plan["kv_bytes_per_token"] == 28672
    off = asyncio.run(_call("plan_llamacpp_offload", {"model": "qwen2.5-32b", "gguf_gib": 18.5, "gpu_gib": 15.5})).data
    assert 1 < off["ngl"] < off["total_blocks"] and off["layout_source"] == "estimate"
    missing = asyncio.run(_call("plan_llamacpp_offload", {"model": "qwen2.5-32b"}))
    assert missing.is_error
    bad = asyncio.run(
        _call("plan_vllm_deployment", {"model": "qwen2.5-7b", "weights_gib": 5, "kv_cache_dtype": "int3"})
    )
    assert bad.is_error and "kv_cache_dtype" in bad.content[0].text


def test_gpu_status_without_a_driver(monkeypatch):
    monkeypatch.setattr("inference_lab.mcp_server.gpu_snapshot", lambda: None)
    assert "NVML unavailable" in asyncio.run(_call("gpu_status")).data["error"]
    assert gpu.gpu_snapshot is not None


@pytest.fixture
def lab_dir(tmp_path, monkeypatch, server):
    """A working directory whose configs/backends.yaml points 'fake' at the test server."""
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/backends.yaml").write_text(
        f"backends:\n  fake:\n    kind: vllm\n    base_url: {server.url}/v1\n    metrics_url: {server.url}/metrics\n"
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_backend_tools_and_benchmark_through_mcp(lab_dir):
    backends = {b["name"]: b for b in asyncio.run(_call("list_backends")).data}
    assert backends["fake"]["up"] and backends["fake"]["model"] == "fake/qwen-7b"
    assert backends["vllm"]["up"] is False  # nothing on :8000 in CI
    metrics = asyncio.run(_call("backend_metrics", {"backend": "fake"})).data
    assert metrics["cache_config"]["num_gpu_blocks"] == "9011"
    run = asyncio.run(
        _call(
            "run_benchmark",
            {
                "backend": "fake",
                "concurrency": [2],
                "requests_per_level": 4,
                "prompt_tokens": 64,
                "max_tokens": 8,
                "label": "fake-run",
            },
        )
    ).data
    assert run["levels"][0]["output_tokens"] == 32 and (lab_dir / run["saved"]).exists()
    assert "fake-run" in (lab_dir / "results/latest.md").read_text()

    async def read():
        async with Client(mcp) as c:
            return await c.read_resource("lab://results/latest")

    assert "fake-run" in asyncio.run(read())[0].text


def test_stdio_server_starts_as_a_subprocess():
    """What Claude Code does: launch `inference-lab mcp` and talk MCP over stdin/stdout."""
    import sys

    from fastmcp.client.transports import StdioTransport

    async def go():
        transport = StdioTransport(command=sys.executable, args=["-m", "inference_lab.cli", "mcp"])
        async with Client(transport) as c:
            tools = [t.name for t in await c.list_tools()]
            res = await c.call_tool("kv_cache_size", {"model": "llama-3.1-8b", "tokens": 8192})
        return tools, res

    tools, res = asyncio.run(go())
    assert "plan_llamacpp_offload" in tools
    rows = res.structured_content.get("result", res.data) if res.structured_content else res.data
    assert any(r["dtype"] == "float16" and r["kib_per_token"] == 128.0 for r in rows)


def _sweep(label: str, prompt: int, levels: dict[int, dict], **extra) -> dict:
    """A saved benchmark run with the given {concurrency: level fields}."""
    base = {"requests": 48, "errors": 0, "ttft_p95_ms": 900.0, "tpot_p50_ms": 10.0, "output_tokens": 12288}
    return {
        "label": label,
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "started_at": "2026-10-07 13:00 UTC",
        "prompt_tokens": prompt,
        "max_tokens": 256,
        "levels": [{**base, "concurrency": c, "mean_output_tokens": 256.0, **lv} for c, lv in levels.items()],
        "gpu": None,
        "server": {},
        **extra,
    }


@pytest.fixture
def saved_runs(tmp_path, monkeypatch):
    from inference_lab import report

    monkeypatch.setenv("INFERENCE_LAB_HOME", str(tmp_path))
    runs = [
        _sweep("vllm-fp8kv-chat", 512, {1: {"throughput_tok_s": 120.0}, 16: {"throughput_tok_s": 1210.0}}),
        _sweep("vllm-fp16kv-chat", 512, {16: {"throughput_tok_s": 1090.0, "ttft_p95_ms": 700.0}}),
        _sweep("flaky-chat", 512, {16: {"throughput_tok_s": 5000.0, "errors": 3}}),  # fast because it failed
        _sweep("ollama-q4km", 512, {16: {"throughput_tok_s": 99.0, "mean_output_tokens": 180.5}}),
        _sweep("llamacpp-f16kv-chat", 512, {1: {"throughput_tok_s": 128.0}, 4: {"throughput_tok_s": 199.0}}),
        _sweep("vllm-fp8kv-long", 4096, {16: {"throughput_tok_s": 327.0}, 32: {"throughput_tok_s": 351.0}}),
        _sweep("vllm-prefix", 128, {16: {"throughput_tok_s": 9999.0}}, system_tokens=2048, shared_system=True),
    ]
    for run in runs:
        report.save(run, tmp_path / "results")
    return tmp_path


def test_compare_configs_ranks_measured_levels_per_scenario(saved_runs):
    res = asyncio.run(_call("compare_configs", {"streams": 16}))
    assert not res.is_error
    out = res.data
    assert out["streams"] == 16 and out["metric"] == "throughput_tok_s" and out["better"] == "higher"
    chat, long = out["scenarios"]  # never ranked against each other; the prefix run is left out
    assert (chat["prompt_tokens"], long["prompt_tokens"]) == (512, 4096)
    assert chat["best"] == "vllm-fp8kv-chat" and long["best"] == "vllm-fp8kv-long"
    order = [r["label"] for r in chat["ranking"]]
    assert order == ["vllm-fp8kv-chat", "vllm-fp16kv-chat", "ollama-q4km", "flaky-chat"]  # errors rank last
    assert [r["rank"] for r in chat["ranking"]] == [1, 2, 3, 4]
    ollama = chat["ranking"][2]
    assert ollama["note"] == "stopped early: 180.5 of 256 output tokens on average"
    assert "note" not in chat["ranking"][0]
    assert out["not_measured_at_streams"] == [{"label": "llamacpp-f16kv-chat", "prompt_tokens": 512, "levels": [1, 4]}]
    assert all(r["label"] != "vllm-prefix" for s in out["scenarios"] for r in s["ranking"])


def test_compare_configs_other_metrics_and_prompt_filter(saved_runs):
    ttft = asyncio.run(_call("compare_configs", {"streams": 16, "metric": "ttft_p95_ms", "prompt_tokens": 512})).data
    assert ttft["better"] == "lower" and len(ttft["scenarios"]) == 1
    assert ttft["scenarios"][0]["best"] == "vllm-fp16kv-chat"  # 700 ms beats 900 ms

    from inference_lab import report

    lab = saved_runs
    energy = _sweep("vllm-fp8kv-chat", 512, {16: {"throughput_tok_s": 1250.0, "energy_j": 4096.0}})
    energy["started_at"] = "2026-10-31 13:00 UTC"
    report.save(energy, lab / "results")
    tpj = asyncio.run(_call("compare_configs", {"streams": 16, "metric": "tokens_per_joule"})).data
    (chat,) = tpj["scenarios"]
    assert chat["best"] == "vllm-fp8kv-chat" and chat["ranking"][0]["tokens_per_joule"] == 3.0  # 12288 / 4096
    assert "vllm-fp16kv-chat" in tpj["missing_metric"] and "vllm-fp8kv-long" in tpj["missing_metric"]


def test_compare_configs_errors(saved_runs, tmp_path_factory, monkeypatch):
    unmeasured = asyncio.run(_call("compare_configs", {"streams": 10}))
    assert unmeasured.is_error and "measured levels: [1, 4, 16, 32]" in unmeasured.content[0].text
    size = asyncio.run(_call("compare_configs", {"streams": 16, "prompt_tokens": 2048}))
    assert size.is_error and "saved prompt sizes: [512, 4096]" in size.content[0].text
    nojoules = asyncio.run(_call("compare_configs", {"streams": 32, "metric": "tokens_per_joule"}))
    assert nojoules.is_error and "none recorded tokens_per_joule" in nojoules.content[0].text
    assert asyncio.run(_call("compare_configs", {"streams": 0})).is_error  # schema: minimum 1
    assert asyncio.run(_call("compare_configs", {"streams": 16, "metric": "vibes"})).is_error  # schema: enum

    monkeypatch.setenv("INFERENCE_LAB_HOME", str(tmp_path_factory.mktemp("empty")))
    empty = asyncio.run(_call("compare_configs", {"streams": 16}))
    assert empty.is_error and "No saved benchmark runs" in empty.content[0].text


def test_compare_configs_schema_is_explicit():
    async def go():
        async with Client(mcp) as c:
            return next(t for t in await c.list_tools() if t.name == "compare_configs")

    tool = asyncio.run(go())
    schema = tool.input_schema if hasattr(tool, "input_schema") else tool.inputSchema
    props = schema["properties"]
    assert schema["required"] == ["streams"] and props["streams"]["minimum"] == 1
    assert "tokens_per_joule" in props["metric"]["enum"] and props["metric"]["default"] == "throughput_tok_s"
    assert all(p.get("description") for p in props.values())
    assert "fastest under N concurrent streams" in tool.description
