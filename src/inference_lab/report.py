"""Save benchmark runs as JSON and build one comparison table across all of them."""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import paths


def save(result: dict, results_dir: Path | None = None) -> Path:
    results_dir = results_dir or paths.results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = re.sub(r"[^0-9]", "", result["started_at"])[:12]
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", result["label"]).strip("-")
    path = results_dir / f"{slug}-{stamp}.json"
    path.write_text(json.dumps(result, indent=2))
    return path


def load_all(results_dir: Path | None = None) -> list[dict]:
    """Latest run per label, oldest label first."""
    results_dir = results_dir or paths.results_dir()
    latest: dict[str, dict] = {}
    for path in sorted(results_dir.glob("*.json")):
        run = json.loads(path.read_text())
        if "levels" in run:
            latest[run["label"]] = run
    return list(latest.values())


def _fmt(x, suffix: str = "", nd: int | None = None) -> str:
    if x is None:
        return "–"
    if nd is not None:
        x = round(x, nd)
    return f"{x:g}{suffix}" if isinstance(x, float) else f"{x}{suffix}"


def _pct(x) -> str:
    return "–" if x is None else f"{x:.0%}"


def markdown(runs: list[dict]) -> str:
    if not runs:
        return "# Benchmarks\n\nNo runs yet. Start a server, then `inference-lab bench --backend vllm`.\n"
    gpu = next((r["gpu"] for r in runs if r.get("gpu")), None)
    lines = ["# Benchmarks", ""]
    if gpu:
        lines.append(f"GPU: {gpu['name']}, {gpu['memory_total_gib']} GiB, driver {gpu['driver']}.")
    lines += [
        "TTFT = time to first token; TPOT = time per output token after that. Prompt / output = mean tokens per "
        "request as the server counted them (chat template included); target in brackets.",
        "",
        "| Server config | Model | Concurrency | Prompt / output tokens | Throughput (tok/s) "
        "| TTFT p50 / p95 (ms) | TPOT p50 (ms) | Peak VRAM (GiB) | Peak KV cache | Preemptions | Errors |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        target = f"[{run.get('prompt_tokens', '?')} / {run.get('max_tokens', '?')}]"
        for lv in run["levels"]:
            tokens = f"{_fmt(lv.get('mean_prompt_tokens'))} / {_fmt(lv.get('mean_output_tokens'))} {target}"
            lines.append(
                f"| {run['label']} | {run['model']} | {lv['concurrency']} | {tokens} | {_fmt(lv['throughput_tok_s'])} "
                f"| {_fmt(lv['ttft_p50_ms'])} / {_fmt(lv['ttft_p95_ms'])} | {_fmt(lv['tpot_p50_ms'])} "
                f"| {_fmt(lv['peak_vram_gib'])} | {_pct(lv['peak_kv_cache_usage'])} | {_fmt(lv['preemptions'])} "
                f"| {lv['errors']}/{lv['requests']} |"
            )
    lines += [
        "",
        "Peak KV cache comes from vLLM's `vllm:kv_cache_usage_perc`; current llama.cpp and Ollama don't report it (–).",
    ]
    alloc = [(r["label"], r["server"]["cache_config"]) for r in runs if r.get("server", {}).get("cache_config")]
    if alloc:
        lines += [
            "",
            "## KV cache vLLM allocated",
            "",
            "| Server config | KV dtype | GPU blocks | Tokens of KV cache |",
        ]
        lines.append("| --- | --- | ---: | ---: |")
        for label, cfg in alloc:
            blocks = int(float(cfg.get("num_gpu_blocks", 0) or 0))
            size = int(float(cfg.get("block_size", 16) or 16))
            lines.append(f"| {label} | {cfg.get('cache_dtype', '?')} | {blocks:,} | {blocks * size:,} |")
        lines.append("")
        lines.append("Compare with `inference-lab plan vllm` for the same model, context and memory settings.")
    return "\n".join(lines) + "\n"


def write_markdown(results_dir: Path | None = None) -> Path:
    results_dir = results_dir or paths.results_dir()
    path = results_dir / "latest.md"
    results_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown(load_all(results_dir)))
    return path
