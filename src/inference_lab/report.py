"""Save benchmark runs as JSON and build one comparison table and SVG charts across all of them."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from xml.sax.saxutils import escape

from . import paths

# file name -> (level field, chart title, factor applied to the field, log y-axis)
CHARTS = {
    "throughput.svg": ("throughput_tok_s", "Throughput (output tokens/s, all streams)", 1.0, False),
    "ttft-p95.svg": ("ttft_p95_ms", "Time to first token, p95 (seconds, log scale)", 0.001, True),
}
# One color per server config, in this order; the README table carries the exact values.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def save(result: dict, results_dir: Path | None = None) -> Path:
    results_dir = results_dir or paths.results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = re.sub(r"[^0-9]", "", result["started_at"])[:12]
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", result["label"]).strip("-")
    path = results_dir / f"{slug}-{stamp}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path


def load_all(results_dir: Path | None = None) -> list[dict]:
    """Latest run per label, oldest label first."""
    results_dir = results_dir or paths.results_dir()
    latest: dict[str, dict] = {}
    for path in sorted(results_dir.glob("*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
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
    path.write_text(markdown(load_all(results_dir)), encoding="utf-8")
    return path


def _config(label: str) -> str:
    """Server config without run_matrix.sh's scenario suffix, so a config keeps one color in every panel."""
    return re.sub(r"-(chat|long)$", "", label)


def _ticks(lo: float, hi: float, log: bool) -> list[float]:
    if log:
        first = math.floor(math.log10(lo))
        return [10.0**e for e in range(first, max(math.ceil(math.log10(hi)), first + 1) + 1)]
    step = 10.0 ** math.floor(math.log10(hi))
    step = next(step * m for m in (0.2, 0.5, 1, 2) if hi / (step * m) <= 5)
    return [step * i for i in range(math.ceil(hi / step) + 1)]


def _panel(group: list[dict], key: str, scale: float, log: bool, x0: int, color: dict[str, str]) -> list[str]:
    """One plot (all runs with the same prompt size): a line per server config over concurrency."""
    width, height, top = 330, 220, 96
    levels = sorted({lv["concurrency"] for r in group for lv in r["levels"]})
    values = [lv[key] * scale for r in group for lv in r["levels"] if lv.get(key)]
    if not values:
        return []
    ticks = _ticks(min(values), max(values), log)

    def px(concurrency: int) -> float:
        return round(x0 + 20 + (width - 40) * levels.index(concurrency) / max(1, len(levels) - 1), 1)

    def py(v: float) -> float:
        frac = math.log10(v / ticks[0]) / math.log10(ticks[-1] / ticks[0]) if log else v / ticks[-1]
        return round(top + height * (1 - frac), 1)

    tokens = group[0].get("prompt_tokens")
    out = [f'<text x="{x0}" y="{top - 12}" font-weight="600" fill="{INK}">{tokens:,}-token prompts</text>']
    for t in ticks:
        out.append(f'<path d="M{x0} {py(t)}h{width}" stroke="{GRID}"/>')
        out.append(f'<text x="{x0 - 6}" y="{py(t) + 4}" text-anchor="end" fill="{MUTED}">{t:,g}</text>')
    for c in levels:
        out.append(f'<text x="{px(c)}" y="{top + height + 18}" text-anchor="middle" fill="{MUTED}">{c}</text>')
    out.append(
        f'<text x="{x0 + width / 2}" y="{top + height + 38}" text-anchor="middle" fill="{MUTED}">'
        "Concurrent requests</text>"
    )
    for run in group:
        stroke = color[_config(run["label"])]
        pts = [(px(lv["concurrency"]), py(lv[key] * scale)) for lv in run["levels"] if lv.get(key)]
        line = " ".join(f"{x},{y}" for x, y in pts)
        out.append(
            f'<polyline points="{line}" fill="none" stroke="{stroke}" stroke-width="2" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        out += [
            f'<circle cx="{x}" cy="{y}" r="4" fill="{stroke}" stroke="{SURFACE}" stroke-width="2"/>' for x, y in pts
        ]
    return out


def chart_svg(runs: list[dict], key: str, title: str, scale: float = 1.0, log: bool = False) -> str:
    """Line chart of one level field against concurrency, one panel per prompt size. Standard library only."""
    configs = sorted({_config(r["label"]) for r in runs})
    color = {c: PALETTE[i] if i < len(PALETTE) else MUTED for i, c in enumerate(configs)}
    sizes = sorted({r.get("prompt_tokens", 0) for r in runs})
    left, pitch = 56, 394
    width, height = left + pitch * len(sizes), 368
    days = sorted({r["started_at"][:10] for r in runs})
    gpu = next((r["gpu"]["name"] for r in runs if r.get("gpu")), None)
    sub = " · ".join(filter(None, [gpu, "runs " + " to ".join(dict.fromkeys([days[0], days[-1]]))]))
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        'role="img" font-family="system-ui, -apple-system, Segoe UI, sans-serif" font-size="12">',
        f"<title>{escape(title)}</title>",
        f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>',
        f'<text x="{left}" y="24" font-size="15" font-weight="600" fill="{INK}">{escape(title)}</text>',
        f'<text x="{left}" y="42" fill="{MUTED}">{escape(sub)}</text>',
    ]
    x = left
    for c in configs:  # legend
        out.append(
            f'<path d="M{x} 62h20" stroke="{color[c]}" stroke-width="2"/>'
            f'<circle cx="{x + 10}" cy="62" r="4" fill="{color[c]}"/>'
            f'<text x="{x + 26}" y="66" fill="{INK}">{escape(c)}</text>'
        )
        x += 40 + 7 * len(c)
    for i, size in enumerate(sizes):
        out += _panel([r for r in runs if r.get("prompt_tokens", 0) == size], key, scale, log, left + pitch * i, color)
    return "\n".join(out) + "\n</svg>\n"


def write_charts(results_dir: Path | None = None) -> list[Path]:
    results_dir = results_dir or paths.results_dir()
    runs = load_all(results_dir)
    written = []
    for name, (key, title, scale, log) in CHARTS.items() if runs else ():
        path = results_dir / name
        path.write_text(chart_svg(runs, key, title, scale, log), encoding="utf-8")
        written.append(path)
    return written
