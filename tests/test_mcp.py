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
