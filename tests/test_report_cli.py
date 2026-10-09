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


def test_charts_draw_a_line_per_config_in_a_panel_per_prompt_size(tmp_path):
    from xml.etree import ElementTree

    report.save(_run("vllm-fp16kv-chat", "2026-10-06 18:00 UTC", 400.0), tmp_path)
    long = _run("vllm-fp16kv-long", "2026-10-07 18:00 UTC", 300.0)
    long["prompt_tokens"] = 4096
    report.save(long, tmp_path)
    report.save(_run("ollama-q4km", "2026-10-06 19:00 UTC", 100.0), tmp_path)
    charts = report.write_charts(tmp_path)
    assert [p.name for p in charts] == ["throughput.svg", "ttft-p95.svg"]
    svg = charts[0].read_text(encoding="utf-8")
    ElementTree.fromstring(svg)  # well-formed XML
    assert svg.count("<polyline") == 3 and "512-token prompts" in svg and "4,096-token prompts" in svg
    assert ">vllm-fp16kv<" in svg and ">ollama-q4km<" in svg and "-chat<" not in svg
    assert "RTX 5070 Ti · runs 2026-10-06 to 2026-10-07" in svg
    empty = tmp_path / "empty"
    empty.mkdir()
    assert report.write_charts(empty) == []


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


def test_files_are_utf8_whatever_the_platform_default(tmp_path):
    # Windows defaults to cp1252, which turned the report's dashes into invalid UTF-8 on GitHub.
    # -X warn_default_encoding makes any read or write that relies on the default an error.
    import subprocess
    import sys

    code = (
        "import sys; from pathlib import Path; from inference_lab import report;"
        "from tests.test_report_cli import _run;"
        "d = Path(sys.argv[1]); report.save(_run('a', '2026-10-06 18:00 UTC', 1.0), d);"
        "report.write_markdown(d)"
    )
    subprocess.run(
        [sys.executable, "-X", "warn_default_encoding", "-W", "error::EncodingWarning", "-c", code, str(tmp_path)],
        check=True,
    )
    assert "–" in (tmp_path / "latest.md").read_bytes().decode("utf-8")
