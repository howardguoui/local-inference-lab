"""inference-lab: plan, benchmark and report on local LLM serving.

inference-lab gpu
inference-lab plan kv --model qwen2.5-7b --tokens 32768
inference-lab plan vllm --model qwen2.5-7b --weights-gib 5.2 --max-model-len 8192 --kv-cache-dtype fp8
inference-lab plan vllm --model qwen2.5-7b --weights-gib 5.19 --vllm-log logs/vllm-fp16kv.log
inference-lab plan llamacpp --gguf-path models/Qwen2.5-32B-Instruct-Q4_K_M.gguf --ctx 8192 -ctk q8_0 -ctv q8_0
inference-lab bench --backend vllm --concurrency 1 4 16 --label vllm-fp16kv
inference-lab bench --backend vllm --system-tokens 2048 --shared-system --prompt-tokens 128 --label vllm-prefix
inference-lab accuracy --backend vllm --label vllm-fp8kv
inference-lab report
inference-lab demo
inference-lab mcp
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from . import report
from .backends import load_backends
from .gguf_layout import estimate_layout, read_gguf
from .gpu import default_gpu_gib, gpu_snapshot
from .models import get_model
from .planner import LLAMACPP_KV_BYTES, VLLM_KV_BYTES, calibrate_vllm, kv_dtype_table, plan_llamacpp, plan_vllm
from .vllm_log import read_vllm_log


def _size_gib(path: str) -> float:
    """Size of a model file, or of a checkpoint directory's weights in one format
    (safetensors, else GGUF, else .bin, so a folder holding two formats isn't double-counted)."""
    p = Path(path)
    if p.is_file():
        return p.stat().st_size / 1024**3
    for suffix in (".safetensors", ".gguf", ".bin"):
        files = list(p.rglob(f"*{suffix}"))
        if files:
            return sum(f.stat().st_size for f in files) / 1024**3
    raise SystemExit(f"No model files found at {path}")


def _gpu(arg: float | None, which: str) -> float:
    if arg is not None:
        return arg
    gib, source = default_gpu_gib(which)
    print(f"  planning against {source}\n")
    return gib


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
    w.add_argument("--weights-path", help="checkpoint directory; sums its weight files")
    pv.add_argument("--max-model-len", type=int, help="default: the log's value with --vllm-log, else 8192")
    pv.add_argument("--gpu-gib", type=float, help="default: total VRAM from NVML")
    pv.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    pv.add_argument("--kv-cache-dtype", default="auto", choices=sorted(VLLM_KV_BYTES))
    pv.add_argument("--overhead-gib", type=float, default=1.5, help="activations + CUDA graphs (estimate)")
    pv.add_argument("--vllm-log", help="startup log of a vLLM server run with these settings: compare the plan with it")

    pl = plan.add_parser("llamacpp", help="largest -ngl that fits, for full or partial GPU offload")
    g = pl.add_mutually_exclusive_group(required=True)
    g.add_argument("--gguf-path", help="read exact tensor sizes and the model shape from the file")
    g.add_argument("--gguf-gib", type=float, help="file size only (needs --model); layout is estimated")
    pl.add_argument("--model", help="preset or config.json; with --gguf-path the file's own metadata is used")
    pl.add_argument("--ctx", type=int, default=8192, help="total context (-c), shared by all -np slots")
    pl.add_argument("--gpu-gib", type=float, help="default: free VRAM from NVML")
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
    b.add_argument("--allow-eos", action="store_true", help="let the model stop early (default: fixed length)")
    b.add_argument("--config", help="backends YAML (default configs/backends.yaml)")
    b.add_argument(
        "--system-tokens", type=int, default=0, help="prefix-caching scenario: a system prompt of about this length"
    )
    b.add_argument(
        "--shared-system",
        action="store_true",
        help="send the same system prompt every time (default: unique per request, the no-reuse control)",
    )

    acc = sub.add_parser("accuracy", help="answer accuracy on fixed arithmetic word problems (exact match)")
    acc.add_argument("--backend", required=True, help="name from configs/backends.yaml: vllm, llamacpp, ollama")
    acc.add_argument("--label", help="server config this run belongs to, e.g. vllm-fp8kv (default: the backend)")
    acc.add_argument("--problems", type=int, default=50, help="how many problems (default 50)")
    acc.add_argument("--seed", type=int, default=0, help="problem set seed; keep 0 so runs stay comparable")
    acc.add_argument("--concurrency", type=int, default=8, help="requests in flight")
    acc.add_argument("--max-tokens", type=int, default=512, help="room for the reasoning before the answer")
    acc.add_argument("--config", help="backends YAML (default configs/backends.yaml)")

    sub.add_parser("report", help="rebuild results/latest.md and the SVG charts from saved runs")

    d = sub.add_parser("demo", help="build the static demo page (GitHub Pages) from saved runs")
    d.add_argument("--out", default="docs", help="folder GitHub Pages serves")

    m = sub.add_parser("mcp", help="run the MCP server (stdio by default)")
    m.add_argument("--http", action="store_true")
    m.add_argument("--port", type=int, default=8765)

    a = ap.parse_args(argv)

    if a.cmd == "gpu":
        print(json.dumps(gpu_snapshot() or {"error": "NVML unavailable: no NVIDIA driver"}, indent=2))
    elif a.cmd == "demo":
        from .demo import build

        page = build(Path(a.out))
        print(f"Wrote {page} and {page.parent / 'data.json'}")
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
        startup = None
        if a.vllm_log:
            try:
                startup = read_vllm_log(a.vllm_log)
            except (OSError, ValueError) as e:
                raise SystemExit(f"--vllm-log {a.vllm_log}: {e}") from e
        max_len = a.max_model_len or (startup and startup.max_model_len) or 8192
        weights = a.weights_gib if a.weights_gib is not None else _size_gib(a.weights_path)
        plan = plan_vllm(
            get_model(a.model),
            _gpu(a.gpu_gib, "total"),
            weights,
            max_len,
            a.gpu_memory_utilization,
            a.kv_cache_dtype,
            a.overhead_gib,
        )
        _print_plan(plan.as_dict())
        if startup:
            print(f"\n  against {a.vllm_log}:\n")
            _print_plan(calibrate_vllm(plan, startup).as_dict())
    elif a.cmd == "plan" and a.kind == "llamacpp":
        if a.gguf_path:
            layout = read_gguf(a.gguf_path)
            model = get_model(a.model) if a.model else layout.spec
        else:
            if not a.model:
                raise SystemExit("--gguf-gib needs --model")
            model = get_model(a.model)
            layout = estimate_layout(model, int(a.gguf_gib * 1024**3))
        gpu = _gpu(a.gpu_gib, "free")
        _print_plan(plan_llamacpp(model, layout, a.ctx, gpu, a.cache_type_k, a.cache_type_v, a.reserve_gib).as_dict())
    elif a.cmd == "bench":
        from .bench import run_benchmark

        if a.shared_system and a.system_tokens <= 0:
            raise SystemExit("--shared-system needs --system-tokens, e.g. --system-tokens 2048")
        backend = load_backends(a.config)[a.backend]
        result = asyncio.run(
            run_benchmark(
                backend,
                a.concurrency,
                a.requests,
                a.prompt_tokens,
                a.max_tokens,
                a.label,
                ignore_eos=not a.allow_eos,
                system_tokens=a.system_tokens,
                shared_system=a.shared_system,
            )
        ).as_dict()
        path = report.save(result)
        print(report.markdown([result]))
        print(f"Saved {path}; combined table in {report.write_markdown()}")
    elif a.cmd == "accuracy":
        from .accuracy import run_accuracy

        backend = load_backends(a.config)[a.backend]
        try:
            result = asyncio.run(
                run_accuracy(backend, a.problems, a.concurrency, a.max_tokens, a.label, seed=a.seed)
            ).as_dict()
        except ValueError as e:
            raise SystemExit(str(e)) from e
        path = report.save(result)
        print("\n".join(report.accuracy_markdown([result])).strip())
        if result["truncated"]:
            print(f"\n{result['truncated']} replies hit --max-tokens {a.max_tokens}; raise it to score them fairly.")
        print(f"\nSaved {path}; combined table in {report.write_markdown()}")
    elif a.cmd == "report":
        print(report.write_markdown().read_text(encoding="utf-8"))
        report.write_charts()
    elif a.cmd == "mcp":
        from .mcp_server import main as serve

        serve(http=a.http, port=a.port)


if __name__ == "__main__":
    main()
