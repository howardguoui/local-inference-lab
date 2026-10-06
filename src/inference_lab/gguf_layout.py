"""Where a GGUF model's bytes are: token embeddings, output head, and each repeating block.

llama.cpp places these separately (see `plan_llamacpp`), so an even split of the file
size over the layers is wrong by the size of the vocabulary tensors, which for
Qwen-style 150k-token vocabularies is hundreds of MiB. Reading the file's tensor table
gives exact sizes; `estimate_layout` is the fallback when only the file size is known.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .models import ModelSpec

_BLOCK = re.compile(r"^blk\.(\d+)\.")


@dataclass
class GgufLayout:
    n_layers: int
    token_embd_bytes: int
    output_bytes: int  # the output head; for tied embeddings llama.cpp reuses a copy of token_embd
    block_bytes: list[int]  # per repeating block, index = layer
    other_bytes: int = 0  # norms and small tensors outside the blocks
    source: str = "gguf"
    spec: ModelSpec | None = field(default=None, repr=False)

    @property
    def total_bytes(self) -> int:
        return self.token_embd_bytes + self.output_bytes + sum(self.block_bytes) + self.other_bytes


def _field(reader, key: str):
    f = reader.fields.get(key)
    if f is None:
        return None
    values = [f.parts[i] for i in f.data]
    if len(values) == 1 and len(values[0]) == 1:
        v = values[0][0]
        return v.item() if hasattr(v, "item") else v
    return [int(v[0]) for v in values]  # per-layer arrays (e.g. head_count_kv on some models)


def read_gguf(path: str | Path) -> GgufLayout:
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    arch = bytes(reader.fields["general.architecture"].parts[-1]).decode()
    n_layers = int(_field(reader, f"{arch}.block_count"))
    blocks = [0] * n_layers
    embd = out = other = 0
    vocab = 0
    for t in reader.tensors:
        m = _BLOCK.match(t.name)
        if m and int(m.group(1)) < n_layers:
            blocks[int(m.group(1))] += int(t.n_bytes)
        elif t.name == "token_embd.weight":
            embd = int(t.n_bytes)
            vocab = int(max(t.shape))
        elif t.name == "output.weight":
            out = int(t.n_bytes)
        else:
            other += int(t.n_bytes)
    tied = out == 0
    spec = None
    heads = _field(reader, f"{arch}.attention.head_count")
    hidden = _field(reader, f"{arch}.embedding_length")
    if heads and hidden:
        heads = max(heads) if isinstance(heads, list) else int(heads)
        kv = _field(reader, f"{arch}.attention.head_count_kv") or heads
        kv = max(kv) if isinstance(kv, list) else int(kv)
        spec = ModelSpec(
            name=Path(path).stem,
            n_layers=n_layers,
            n_heads=heads,
            n_kv_heads=kv,
            head_dim=int(_field(reader, f"{arch}.attention.key_length") or int(hidden) // heads),
            hidden_size=int(hidden),
            vocab_size=vocab,
            tie_embeddings=tied,
        )
    return GgufLayout(
        n_layers=n_layers,
        token_embd_bytes=embd,
        output_bytes=embd if tied else out,  # tied: the GPU still needs its own copy
        block_bytes=blocks,
        other_bytes=other,
        spec=spec,
    )


def estimate_layout(model: ModelSpec, file_bytes: int) -> GgufLayout:
    """Split a file size into embeddings, output head and blocks using the model's shape,
    at the file's average bits per weight. Good to a few percent; read_gguf is exact."""
    vocab_params = model.vocab_size * model.hidden_size
    total_params = model.params_b * 1e9 if model.params_b else None
    bytes_per_param = file_bytes / total_params if total_params else 0.6  # ~4.8 bits/weight (Q4_K_M)
    embd = int(vocab_params * bytes_per_param)
    # llama.cpp's k-quant mixes keep the output head at Q6_K or better (6.5625 bits/weight),
    # above a Q4 file's average, so don't let it fall below that
    out = int(vocab_params * max(bytes_per_param, 6.5625 / 8))
    file_has_output = 0 if model.tie_embeddings else out
    repeat = max(0, file_bytes - embd - file_has_output)
    per_block = repeat // model.n_layers
    return GgufLayout(
        n_layers=model.n_layers,
        token_embd_bytes=embd,
        output_bytes=out,
        block_bytes=[per_block] * model.n_layers,
        source="estimate",
        spec=model,
    )
