"""Static demo for GitHub Pages: the published benchmark runs, plus the vLLM planner running in the browser.

`inference-lab demo` writes docs/ (index.html, planner.js, data.json). Every number on the page comes from
results/*.json (real GPU runs) or from planner.py itself: data.json carries the model presets and planner
constants, and tests check that planner.js gives the same answers as plan_vllm.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import report
from .models import PRESETS
from .planner import GIB, VLLM_KV_BYTES, plan_vllm

STATIC = Path(__file__).parent / "static" / "demo"

# Checkpoints the published runs served: planner preset and the summed size of the weight files on
# Hugging Face (Qwen2.5-7B-Instruct-AWQ: 3,996,422,976 + 1,574,406,784 bytes).
CHECKPOINTS = {"Qwen/Qwen2.5-7B-Instruct-AWQ": {"preset": "qwen2.5-7b", "weights_bytes": 5_570_829_760}}

LEVEL_FIELDS = (
    "concurrency",
    "requests",
    "errors",
    "throughput_tok_s",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "tpot_p50_ms",
    "peak_vram_gib",
    "peak_kv_cache_usage",
    "preemptions",
    "mean_prompt_tokens",
)


def summarize_run(run: dict) -> dict:
    """The part of a saved run the page shows."""
    cache = (run.get("server") or {}).get("cache_config") or {}
    return {
        "label": run["label"],
        "backend": (run.get("backend") or {}).get("kind") or (run.get("backend") or {}).get("name"),
        "model": run.get("model"),
        "scenario": "long" if (run.get("prompt_tokens") or 0) > 1024 else "chat",
        "prompt_tokens": run.get("prompt_tokens"),
        "max_tokens": run.get("max_tokens"),
        "started_at": run.get("started_at"),
        "gpu": (run.get("gpu") or {}).get("name"),
        "gpu_gib": (run.get("gpu") or {}).get("memory_total_gib"),
        "kv_cache": {
            "dtype": cache.get("cache_dtype"),
            "blocks": int(cache["num_gpu_blocks"]) if cache.get("num_gpu_blocks") else None,
            "block_size": int(cache["block_size"]) if cache.get("block_size") else None,
            "gpu_memory_utilization": float(cache["gpu_memory_utilization"])
            if cache.get("gpu_memory_utilization")
            else None,
        },
        "levels": [{k: lv.get(k) for k in LEVEL_FIELDS} for lv in run.get("levels", [])],
    }


def calibration(runs: list[dict], max_model_len: int = 8192) -> list[dict]:
    """Planner prediction vs the KV cache vLLM actually allocated, once per KV dtype."""
    rows, seen = [], set()
    for run in runs:
        kv = run["kv_cache"]
        ckpt = CHECKPOINTS.get(run["model"])
        if not kv["blocks"] or not ckpt or kv["dtype"] in seen:
            continue
        seen.add(kv["dtype"])
        dtype = "fp8" if str(kv["dtype"]).startswith("fp8") else "auto"
        plan = plan_vllm(
            PRESETS[ckpt["preset"]],
            gpu_gib=run["gpu_gib"],
            weights_gib=ckpt["weights_bytes"] / GIB,
            max_model_len=max_model_len,
            gpu_memory_utilization=kv["gpu_memory_utilization"] or 0.9,
            kv_cache_dtype=dtype,
        )
        actual = kv["blocks"] * kv["block_size"]
        rows.append(
            {
                "model": run["model"],
                "preset": ckpt["preset"],
                "kv_cache_dtype": dtype,
                "gpu_gib": run["gpu_gib"],
                "weights_gib": round(ckpt["weights_bytes"] / GIB, 3),
                "gpu_memory_utilization": plan.gpu_memory_utilization,
                "predicted_tokens": plan.kv_tokens,
                "actual_tokens": actual,
                "error_pct": round(100 * (plan.kv_tokens - actual) / actual, 1),
            }
        )
    return sorted(rows, key=lambda r: r["kv_cache_dtype"])


def planner_spec() -> dict:
    """Everything planner.js needs to reproduce plan_vllm."""
    defaults = plan_vllm.__defaults__  # (gpu_memory_utilization, kv_cache_dtype, overhead_gib, block_size)
    return {
        "gib": GIB,
        "kv_bytes": {k: v for k, v in VLLM_KV_BYTES.items() if k in ("auto", "fp8")},
        "gpu_memory_utilization": defaults[0],
        "overhead_gib": defaults[2],
        "block_size": defaults[3],
        "models": {
            name: {
                "n_layers": m.n_layers,
                "n_kv_heads": m.n_kv_heads,
                "head_dim": m.head_dim,
                "params_b": m.params_b,
            }
            for name, m in PRESETS.items()
        },
    }


def build(out_dir: Path, results_dir: Path | None = None) -> Path:
    runs = [summarize_run(r) for r in report.load_all(results_dir)]
    if not runs:
        raise ValueError("No saved runs in results/; run `inference-lab bench` first.")
    runs.sort(key=lambda r: r["label"])
    data = {
        "gpu": runs[0]["gpu"],
        "gpu_gib": runs[0]["gpu_gib"],
        "run_dates": sorted({r["started_at"] for r in runs if r["started_at"]}),
        "runs": runs,
        "calibration": calibration(runs),
        "planner": planner_spec(),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "planner.js", "app.js"):
        shutil.copyfile(STATIC / name, out_dir / name)
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")
    (out_dir / "data.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    return out_dir / "index.html"
