"""The static demo: data.json from saved runs, and planner.js matching plan_vllm."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from inference_lab import demo
from inference_lab.cli import main
from inference_lab.models import PRESETS
from inference_lab.planner import plan_vllm

ROOT = Path(__file__).resolve().parents[1]


def _run(label, prompt_tokens, cache=None, levels=(4,)):
    return {
        "label": label,
        "backend": {"name": label.split("-")[0], "kind": label.split("-")[0]},
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "started_at": "2026-10-07 13:00 UTC",
        "prompt_tokens": prompt_tokens,
        "max_tokens": 256,
        "gpu": {"name": "NVIDIA GeForce RTX 5070 Ti", "memory_total_gib": 15.92},
        "server": {"cache_config": cache} if cache else {},
        "levels": [
            {
                "concurrency": c,
                "requests": 48,
                "errors": 0,
                "throughput_tok_s": 100.0 * c,
                "ttft_p50_ms": 150.0,
                "ttft_p95_ms": 200.0,
                "tpot_p50_ms": 8.0,
                "peak_vram_gib": 15.6,
                "peak_kv_cache_usage": 0.1,
                "preemptions": 0.0,
                "mean_prompt_tokens": 480.0,
                "first_error": None,
            }
            for c in levels
        ],
    }


def _cache(dtype, blocks):
    return {"num_gpu_blocks": str(blocks), "block_size": "16", "cache_dtype": dtype, "gpu_memory_utilization": "0.9"}


def _write(folder, runs):
    folder.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(runs):
        (folder / f"{r['label']}-{i}.json").write_text(json.dumps(r), encoding="utf-8")
    return folder


def test_build_writes_page_and_data_from_saved_runs(tmp_path):
    results = _write(
        tmp_path / "results",
        [
            _run("vllm-fp16kv-chat", 512, _cache("auto", 9878), levels=(1, 4, 16)),
            _run("vllm-fp8kv-long", 4096, _cache("fp8", 17019), levels=(8, 48)),
            _run("ollama-q4km", 512),
        ],
    )
    page = demo.build(tmp_path / "site", results)
    data = json.loads((page.parent / "data.json").read_text(encoding="utf-8"))
    assert {p.name for p in page.parent.iterdir()} >= {"index.html", "planner.js", "app.js", "data.json", ".nojekyll"}
    runs = {r["label"]: r for r in data["runs"]}
    assert runs["vllm-fp8kv-long"]["scenario"] == "long" and runs["ollama-q4km"]["scenario"] == "chat"
    assert [lv["concurrency"] for lv in runs["vllm-fp16kv-chat"]["levels"]] == [1, 4, 16]
    assert runs["vllm-fp8kv-long"]["kv_cache"] == {
        "dtype": "fp8",
        "blocks": 17019,
        "block_size": 16,
        "gpu_memory_utilization": 0.9,
    }
    cal = {c["kv_cache_dtype"]: c for c in data["calibration"]}
    assert cal["auto"]["actual_tokens"] == 9878 * 16 and cal["fp8"]["actual_tokens"] == 17019 * 16
    expected = plan_vllm(PRESETS["qwen2.5-7b"], 15.92, 5_570_829_760 / 1024**3, 8192, 0.9, "fp8").kv_tokens
    assert cal["fp8"]["predicted_tokens"] == expected
    assert cal["fp8"]["error_pct"] == round(100 * (expected - 17019 * 16) / (17019 * 16), 1)
    assert data["planner"]["overhead_gib"] == 1.5 and data["planner"]["block_size"] == 16


def test_build_needs_runs(tmp_path):
    with pytest.raises(ValueError):
        demo.build(tmp_path / "site", _write(tmp_path / "empty", []))


def test_demo_command(tmp_path, monkeypatch):
    monkeypatch.setattr(demo.report, "load_all", lambda results_dir=None: [_run("ollama-q4km", 512)])
    main(["demo", "--out", str(tmp_path / "site")])
    assert (tmp_path / "site" / "data.json").exists()


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node to run planner.js")
def test_planner_js_matches_plan_vllm():
    spec = demo.planner_spec()
    cases = [
        dict(model=m, gpu_gib=g, weights_gib=w, max_model_len=n, gpu_memory_utilization=u, kv_cache_dtype=kv)
        for m in PRESETS
        for g in (12.0, 15.92, 24.0)
        for w in (4.1, 5.188, 9.5, 17.0)
        for n in (4096, 8192, 32768)
        for u in (0.85, 0.9)
        for kv in ("auto", "fp8")
    ]
    script = (
        "const P=require(process.argv[1]);let t='';process.stdin.on('data',d=>t+=d);"
        "process.stdin.on('end',()=>{const {spec,cases}=JSON.parse(t);"
        "console.log(JSON.stringify(cases.map(o=>P.planVllm(spec,o))))})"
    )
    js_path = str(ROOT / "src" / "inference_lab" / "static" / "demo" / "planner.js")
    out = subprocess.run(
        ["node", "-e", script, js_path],
        input=json.dumps({"spec": spec, "cases": cases}),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    got = json.loads(out)
    for case, js in zip(cases, got, strict=True):
        py = plan_vllm(
            PRESETS[case["model"]],
            case["gpu_gib"],
            case["weights_gib"],
            case["max_model_len"],
            case["gpu_memory_utilization"],
            case["kv_cache_dtype"],
        )
        assert (js["kv_tokens"], js["kv_blocks"], js["max_concurrent_at_max_len"], js["fits"]) == (
            py.kv_tokens,
            py.kv_blocks,
            py.max_concurrent_at_max_len,
            py.fits,
        ), case
        assert js["kv_budget_gib"] == pytest.approx(py.kv_budget_gib, abs=0.006), case
        assert js["kv_bytes_per_token"] == py.kv_bytes_per_token
