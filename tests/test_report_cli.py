import json

from inference_lab import report
from inference_lab.cli import main


def _run(label: str, started: str, tput: float, cache: dict | None = None) -> dict:
    level = {
        "concurrency": 4,
        "requests": 32,
        "errors": 0,
        "throughput_tok_s": tput,
        "ttft_p50_ms": 80.0,
        "ttft_p95_ms": 140.0,
        "tpot_p50_ms": 11.2,
        "peak_vram_gib": 14.1,
        "peak_kv_cache_usage": 0.37,
        "preemptions": 0.0,
        "mean_prompt_tokens": 548.0,
        "mean_output_tokens": 256.0,
    }
    return {
        "label": label,
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "started_at": started,
        "prompt_tokens": 512,
        "max_tokens": 256,
        "levels": [level],
        "gpu": {"name": "NVIDIA GeForce RTX 5070 Ti", "memory_total_gib": 15.92, "driver": "580.00"},
        "server": {"cache_config": cache} if cache else {},
    }


def test_report_keeps_latest_run_per_label(tmp_path):
    report.save(_run("vllm-fp16kv", "2026-10-06 18:00 UTC", 400.0), tmp_path)
    report.save(
        _run(
            "vllm-fp16kv",
            "2026-10-06 19:00 UTC",
            410.0,
            {"num_gpu_blocks": "9011", "block_size": "16", "cache_dtype": "auto"},
        ),
        tmp_path,
    )
    report.save(_run("llamacpp-q8kv", "2026-10-06 19:30 UTC", 250.0), tmp_path)
    runs = report.load_all(tmp_path)
    assert [r["label"] for r in runs] == ["llamacpp-q8kv", "vllm-fp16kv"]
    md = report.write_markdown(tmp_path).read_text()
    assert (
        "| vllm-fp16kv | Qwen/Qwen2.5-7B-Instruct-AWQ | 4 | 548 / 256 [512 / 256] | 410 "
        "| 80 / 140 | 11.2 | 14.1 | 37% | 0 | 0/32 |" in md
    )
    assert "| vllm-fp16kv | auto | 9,011 | 144,176 |" in md
    assert "RTX 5070 Ti" in md
    assert "No runs yet" in report.markdown([])


def test_cli_plans(capsys, tmp_path):
    from tests.conftest import make_gguf

    main(["plan", "kv", "--model", "llama-3.1-8b", "--tokens", "8192"])
    assert "128.0" in capsys.readouterr().out
    main(["plan", "vllm", "--model", "qwen2.5-7b", "--weights-gib", "5.2", "--max-model-len", "32768"])
    out = capsys.readouterr().out  # no GPU in CI: falls back to 16 GiB and says so
    assert "no NVML" in out and "144,176" in out and "fp8 would hold" in out
    main(["plan", "llamacpp", "--gguf-path", str(make_gguf(tmp_path / "m.gguf")), "--gpu-gib", "8", "-ctk", "q8_0"])
    out = capsys.readouterr().out
    assert "layout_source" in out and "gguf" in out and "-ngl 99" in out
    main(["plan", "llamacpp", "--model", "qwen2.5-32b", "--gguf-gib", "18.5", "--gpu-gib", "15.5"])
    assert "system RAM" in capsys.readouterr().out
    main(["gpu"])
    assert isinstance(json.loads(capsys.readouterr().out), dict)
