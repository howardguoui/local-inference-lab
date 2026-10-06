"""inference-lab: plan, benchmark and report on local LLM serving.

inference-lab gpu
inference-lab plan kv --model qwen2.5-7b --tokens 32768
inference-lab plan vllm --model qwen2.5-7b --weights-gib 5.2 --max-model-len 8192 --kv-cache-dtype fp8
inference-lab plan llamacpp --model qwen2.5-32b --gguf-gib 18.5 --ctx 8192 -ctk q8_0 -ctv q8_0
inference-lab bench --backend vllm --concurrency 1 4 16 --label vllm-fp16kv
inference-lab report
inference-lab mcp
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from . import report
from .backends import load_backends
from .gpu import gpu_snapshot
from .models import get_model
from .planner import LLAMACPP_KV_BYTES, VLLM_KV_BYTES, kv_dtype_table, plan_llamacpp, plan_vllm


def _size_gib(path: str) -> float:
    """Size of a model file, or of every weight file in a directory."""
    p = Path(path)
    files = [p] if p.is_file() else [f for f in p.rglob("*") if f.suffix in (".safetensors", ".gguf", ".bin")]
    if not files:
        raise SystemExit(f"No model files found at {path}")
    return sum(f.stat().st_size for f in files) / 1024**3


def _print_plan(d: dict) -> None:
    width = max(len(k) for k in d)
    for k, v in d.items():
        if k != "advice":
            print(
                f"  {k.ljust(width)}  {v:,}"
                if isinstance(v, int) and not isinstance(v, bool)
                else f"  {k.ljust(width)}  {v}"
            )
    print(f"\n  {d['advice']}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="inference-lab", description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("gpu", help="show the GPU's memory, utilization and power")

    plan = sub.add_parser("plan", help="size a deployment before starting it").add_subparsers(
        dest="kind", required=True
    )
    kv = plan.add_parser("kv", help="KV cache size of one sequence for every storage type")
    kv.add_argument("--model", required=True, help="preset name or path to config.json")
    kv.add_argument("--tokens", type=int, default=8192)

    pv = plan.add_parser("vllm", help="KV cache vLLM will allocate, and how many sequences fit")
    pv.add_argument("--model", required=True)
    w = pv.add_mutually_exclusive_group(required=True)
    w.add_argument("--weights-gib", type=float)
    w.add_argument("--weights-path", help="checkpoint directory; sums its .safetensors files")
    pv.add_argument("--max-model-len", type=int, default=8192)
    pv.add_argument("--gpu-gib", type=float, default=16.0)
    pv.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    pv.add_argument("--kv-cache-dtype", default="auto", choices=sorted(VLLM_KV_BYTES))
    pv.add_argument("--overhead-gib", type=float, default=1.5, help="activations + CUDA graphs (estimate)")

    pl = plan.add_parser("llamacpp", help="largest -ngl that fits, for full or partial GPU offload")
    pl.add_argument("--model", required=True)
    g = pl.add_mutually_exclusive_group(required=True)
    g.add_argument("--gguf-gib", type=float)
    g.add_argument("--gguf-path")
    pl.add_argument("--ctx", type=int, default=8192, help="total context (-c), shared by all -np slots")
    pl.add_argument("--gpu-gib", type=float, default=16.0)
    pl.add_argument("-ctk", "--cache-type-k", default="f16", choices=sorted(LLAMACPP_KV_BYTES))
    pl.add_argument("-ctv", "--cache-type-v", default="f16", choices=sorted(LLAMACPP_KV_BYTES))
    pl.add_argument("--reserve-gib", type=float, default=1.0, help="compute buffer + display (estimate)")

    b = sub.add_parser("bench", help="benchmark a running server")
    b.add_argument("--backend", required=True, help="name from configs/backends.yaml: vllm, llamacpp, ollama")
    b.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    b.add_argument("--requests", type=int, default=32, help="requests per concurrency level")
    b.add_argument("--prompt-tokens", type=int, default=512)
    b.add_argument("--max-tokens", type=int, default=256)
    b.add_argument("--label", help="name this server config in the report, e.g. vllm-fp8kv")
    b.add_argument("--config", help="backends YAML (default configs/backends.yaml)")

    sub.add_parser("report", help="rebuild results/latest.md from saved runs")

    m = sub.add_parser("mcp", help="run the MCP server (stdio by default)")
    m.add_argument("--http", action="store_true")
    m.add_argument("--port", type=int, default=8765)

    a = ap.parse_args(argv)

    if a.cmd == "gpu":
        print(json.dumps(gpu_snapshot() or {"error": "NVML unavailable: no NVIDIA driver"}, indent=2))
    elif a.cmd == "plan" and a.kind == "kv":
        model = get_model(a.model)
        print(
            f"{model.name}: {model.n_layers} layers, {model.n_kv_heads} KV heads x {model.head_dim} dims "
            f"(GQA {model.gqa_ratio}:1), one {a.tokens:,}-token sequence:\n"
        )
        print(f"  {'engine':10} {'dtype':8} {'KiB/token':>10} {'GiB':>8}")
        for r in kv_dtype_table(model, a.tokens):
            print(f"  {r['engine']:10} {r['dtype']:8} {r['kib_per_token']:>10} {r['gib']:>8}")
    elif a.cmd == "plan" and a.kind == "vllm":
        weights = a.weights_gib if a.weights_gib is not None else _size_gib(a.weights_path)
        _print_plan(
            plan_vllm(
                get_model(a.model),
                a.gpu_gib,
                weights,
                a.max_model_len,
                a.gpu_memory_utilization,
                a.kv_cache_dtype,
                a.overhead_gib,
            ).as_dict()
        )
    elif a.cmd == "plan" and a.kind == "llamacpp":
        gguf = a.gguf_gib if a.gguf_gib is not None else _size_gib(a.gguf_path)
        _print_plan(
            plan_llamacpp(
                get_model(a.model), gguf, a.ctx, a.gpu_gib, a.cache_type_k, a.cache_type_v, a.reserve_gib
            ).as_dict()
        )
    elif a.cmd == "bench":
        from .bench import run_benchmark

        backend = load_backends(a.config)[a.backend]
        result = asyncio.run(
            run_benchmark(backend, a.concurrency, a.requests, a.prompt_tokens, a.max_tokens, a.label)
        ).as_dict()
        path = report.save(result)
        print(report.markdown([result]))
        print(f"Saved {path}; combined table in {report.write_markdown()}")
    elif a.cmd == "report":
        print(report.write_markdown().read_text())
    elif a.cmd == "mcp":
        from .mcp_server import main as serve

        serve(http=a.http, port=a.port)


if __name__ == "__main__":
    main()
