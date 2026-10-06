"""Model architecture facts the memory planner needs.

KV cache size depends only on the attention shape: layers, key/value heads and head
dimension. The embedding and output-head sizes (vocabulary x hidden size) matter for
llama.cpp offload, because those two tensors are placed separately from the layers.
Presets are copied from each model's Hugging Face config.json; any other model can be
loaded from its config.json or read straight from a GGUF file.
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
    hidden_size: int
    vocab_size: int
    params_b: float = 0.0  # billions of parameters
    tie_embeddings: bool = False

    @property
    def gqa_ratio(self) -> int:
        """Query heads per KV head. Grouped-query attention shrinks the KV cache by this factor."""
        return self.n_heads // self.n_kv_heads

    @classmethod
    def from_hf_config(cls, config: dict | str | Path, name: str | None = None) -> ModelSpec:
        if not isinstance(config, dict):
            config = json.loads(Path(config).read_text(encoding="utf-8"))
        cfg = config.get("text_config", config)  # multimodal configs nest the language model
        n_heads = cfg["num_attention_heads"]
        return cls(
            name=name or cfg.get("_name_or_path") or cfg.get("model_type", "model"),
            n_layers=cfg["num_hidden_layers"],
            n_heads=n_heads,
            n_kv_heads=cfg.get("num_key_value_heads", n_heads),
            head_dim=cfg.get("head_dim") or cfg["hidden_size"] // n_heads,
            hidden_size=cfg["hidden_size"],
            vocab_size=cfg.get("vocab_size", 0),
            tie_embeddings=bool(cfg.get("tie_word_embeddings", False)),
        )


PRESETS: dict[str, ModelSpec] = {
    m.name: m
    for m in (
        ModelSpec("qwen2.5-7b", 28, 28, 4, 128, hidden_size=3584, vocab_size=152064, params_b=7.62),
        ModelSpec("qwen2.5-14b", 48, 40, 8, 128, hidden_size=5120, vocab_size=152064, params_b=14.7),
        ModelSpec("qwen2.5-32b", 64, 40, 8, 128, hidden_size=5120, vocab_size=152064, params_b=32.5),
        ModelSpec("qwen3-8b", 36, 32, 8, 128, hidden_size=4096, vocab_size=151936, params_b=8.19),
        ModelSpec("qwen3-14b", 40, 40, 8, 128, hidden_size=5120, vocab_size=151936, params_b=14.8),
        ModelSpec("llama-3.1-8b", 32, 32, 8, 128, hidden_size=4096, vocab_size=128256, params_b=8.03),
        ModelSpec("mistral-7b", 32, 32, 8, 128, hidden_size=4096, vocab_size=32768, params_b=7.25),
    )
}


def get_model(name_or_config: str) -> ModelSpec:
    """A preset name, a config.json path (or its directory), or a .gguf file."""
    if name_or_config in PRESETS:
        return PRESETS[name_or_config]
    path = Path(name_or_config)
    if path.suffix == ".gguf" and path.exists():
        from .gguf_layout import read_gguf

        spec = read_gguf(path).spec
        if spec is None:
            raise ValueError(f"{path} has no attention metadata the planner can read")
        return spec
    if path.is_dir():
        path = path / "config.json"
    if path.exists():
        return ModelSpec.from_hf_config(path, name=path.parent.name)
    raise KeyError(
        f"Unknown model {name_or_config!r}. Presets: {', '.join(PRESETS)}; or pass a config.json or .gguf path."
    )
