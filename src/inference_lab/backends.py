"""Inference servers under test. All three speak the OpenAI chat API, so one client fits all."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import yaml

DEFAULTS = {
    "vllm": {
        "kind": "vllm",
        "base_url": "http://localhost:8000/v1",
        "metrics_url": "http://localhost:8000/metrics",
    },
    "llamacpp": {
        "kind": "llamacpp",
        "base_url": "http://localhost:8081/v1",
        "metrics_url": "http://localhost:8081/metrics",
    },
    "ollama": {
        "kind": "ollama",
        "base_url": "http://localhost:11434/v1",
        "metrics_url": None,
        "model": "qwen2.5:7b-instruct",
    },
}


@dataclass
class Backend:
    name: str
    kind: str
    base_url: str
    metrics_url: str | None = None
    model: str | None = None  # None: ask the server (/v1/models)

    def as_dict(self) -> dict:
        return asdict(self)


def load_backends(path: str | Path | None = None) -> dict[str, Backend]:
    """Built-in defaults, overridden by configs/backends.yaml (or the given file) when present."""
    merged = {k: dict(v) for k, v in DEFAULTS.items()}
    candidate = Path(path) if path else Path("configs/backends.yaml")
    if candidate.exists():
        for name, cfg in (yaml.safe_load(candidate.read_text()) or {}).get("backends", {}).items():
            merged.setdefault(name, {}).update(cfg or {})
    return {name: Backend(name=name, **cfg) for name, cfg in merged.items()}


async def discover_model(client: httpx.AsyncClient) -> str:
    resp = await client.get("/models", timeout=10)
    resp.raise_for_status()
    data = resp.json().get("data") or []
    if not data:
        raise RuntimeError("the server lists no models")
    return data[0]["id"]


async def health(backend: Backend, transport: httpx.AsyncBaseTransport | None = None) -> dict:
    """Is the server up, and which model is it serving?"""
    async with httpx.AsyncClient(base_url=backend.base_url, transport=transport) as client:
        try:
            model = backend.model or await discover_model(client)
            if backend.model:
                (await client.get("/models", timeout=5)).raise_for_status()
            return {"name": backend.name, "up": True, "model": model, "base_url": backend.base_url}
        except Exception as exc:
            return {"name": backend.name, "up": False, "error": f"{type(exc).__name__}: {exc}"[:200]}
