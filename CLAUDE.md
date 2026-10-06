# CLAUDE.md: local-inference-lab

Benchmark and planning toolkit for serving open LLMs on one 16 GB consumer GPU (owner's RTX 5070 Ti).

## Commands

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/python -m pytest -q
```

## Rules

- **Never fabricate results.** `results/` holds only real GPU runs from `scripts/publish_results.sh`.
  README numbers come from those files or from deterministic planner output (CLI runs you can reproduce).
- Cloud sessions have no GPU and usually can't reach Hugging Face; test against `tests/fake_server.py`
  and the fake NVML in `tests/conftest.py`.
- Every change keeps lint and tests green and adds tests for new behavior.
- Verify claims about vLLM / llama.cpp / Ollama flags and metrics against their current source
  (sparse-clone github.com/vllm-project/vllm or ggml-org/llama.cpp) before writing them down.
- One roadmap item per change; move it to "Shipped" in ROADMAP.md with the date when done.
- Commit to `main` with a descriptive message; never force-push.
