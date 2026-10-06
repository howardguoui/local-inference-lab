"""Model architecture facts the memory planner needs.

KV cache size depends only on the attention shape: layers, key/value heads and
head dimension. Presets are copied from each model's Hugging Face config.json;
any other model can be loaded from its own config.json with `from_hf_config`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    name: str
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    params_b: float  # billions of parameters, for weight-size estimates

    @property
    def gqa_ratio(self) -> int:
        """Query heads per KV head. Grouped-query attention shrinks the KV cache by this factor."""
        return self.n_heads // self.n_kv_heads

    @classmethod
    def from_hf_config(cls, config: dict | str | Path, name: str | None = None, params_b: float = 0.0) -> ModelSpec:
        if not isinstance(config, dict):
            config = json.loads(Path(config).read_text())
        cfg = config.get("text_config", config)  # multimodal configs nest the language model
        n_heads = cfg["num_attention_heads"]
        head_dim = cfg.get("head_dim") or cfg["hidden_size"] // n_heads
        return cls(
            name=name or cfg.get("_name_or_path") or cfg.get("model_type", "model"),
            n_layers=cfg["num_hidden_layers"],
            n_heads=n_heads,
            n_kv_heads=cfg.get("num_key_value_heads", n_heads),
            head_dim=head_dim,
            params_b=params_b,
        )


PRESETS: dict[str, ModelSpec] = {
    m.name: m
    for m in (
        ModelSpec("qwen2.5-7b", n_layers=28, n_heads=28, n_kv_heads=4, head_dim=128, params_b=7.62),
        ModelSpec("qwen2.5-14b", n_layers=48, n_heads=40, n_kv_heads=8, head_dim=128, params_b=14.7),
        ModelSpec("qwen2.5-32b", n_layers=64, n_heads=40, n_kv_heads=8, head_dim=128, params_b=32.8),
        ModelSpec("qwen3-8b", n_layers=36, n_heads=32, n_kv_heads=8, head_dim=128, params_b=8.19),
        ModelSpec("qwen3-14b", n_layers=40, n_heads=40, n_kv_heads=8, head_dim=128, params_b=14.8),
        ModelSpec("llama-3.1-8b", n_layers=32, n_heads=32, n_kv_heads=8, head_dim=128, params_b=8.03),
        ModelSpec("mistral-7b", n_layers=32, n_heads=32, n_kv_heads=8, head_dim=128, params_b=7.25),
    )
}


def get_model(name_or_config: str) -> ModelSpec:
    """A preset name, or a path to a config.json."""
    if name_or_config in PRESETS:
        return PRESETS[name_or_config]
    path = Path(name_or_config)
    if path.is_dir():
        path = path / "config.json"
    if path.exists():
        return ModelSpec.from_hf_config(path, name=path.parent.name)
    raise KeyError(f"Unknown model {name_or_config!r}. Presets: {', '.join(PRESETS)}; or pass a config.json path.")
