# Roadmap

Shipped items move to the bottom with the date. Results are only ever committed from real runs on the GPU
machine (`scripts/publish_results.sh`); nothing in `results/` is hand-written.

## Next

- [ ] **Finish the matrix:** the 32B partial-offload run (`OFFLOAD=1 scripts/run_matrix.sh`) and Ollama on the
      long scenario, the two parts the 2026-10-07 run did not cover.
- [ ] **Tune `overhead_gib`:** set the vLLM planner's default from the fitted overheads of the next matrix run's
      saved logs (`inference-lab plan vllm --vllm-log logs/<label>.log`).
- [ ] **Run the prefix caching scenario** on the GPU (`run_matrix.sh` now includes it for vLLM) and summarize the
      TTFT drop and hit rate in the README.
- [ ] **Quality check per config:** a small fixed answer-accuracy set (e.g. 50 arithmetic word problems with exact
      answers) so FP8 KV cache and Q4 quantization show any accuracy cost next to their speed gain.
- [ ] **Speculative decoding:** vLLM n-gram / draft-model speculation on the chat scenario; acceptance rate and
      tokens/s change.
- [ ] **SGLang backend** as a fourth OpenAI-compatible server profile.
- [ ] **Power efficiency:** tokens per joule per config from the NVML power samples.
- [ ] **MCP:** a `compare_configs` tool that answers "which config is fastest under N streams" from saved runs.

## Shipped

- 2026-10-19: prefix caching scenario: `inference-lab bench --system-tokens N [--shared-system]` puts a system
  prompt of about N tokens in front of every request, shared or (the control) unique per request, and records each
  level's prefix cache hit rate from the change in vLLM's `prefix_cache_hits` / `prefix_cache_queries` counters.
  `run_matrix.sh` runs the shared/unique pair on the vLLM configs; `results/latest.md` gets a "Prefix caching"
  table. Numbers come with the next GPU run (item above).
- 2026-10-12: planner calibration against vLLM's startup log: `inference-lab plan vllm --vllm-log` parses the
  loaded weights, available KV memory, KV token capacity and CUDA graph memory, and reports predicted vs actual
  KV tokens with the error in percent and the `--overhead-gib` that reproduces that server. `run_matrix.sh` keeps
  each server's log in `logs/`. Tuning the default from real logs is the follow-up item above.
- 2026-10-09: first benchmark matrix on the RTX 5070 Ti published (`results/latest.md`: vLLM and llama.cpp from
  2026-10-07, Ollama chat from 2026-10-06) with the findings summarized in the README.
- 2026-10-09: charts of throughput and TTFT p95 against concurrency per server config, written as SVG from
  `results/*.json` by `inference-lab report` (standard library instead of matplotlib: no new dependency).
- 2026-10-07: demo page on GitHub Pages (`inference-lab demo` → `docs/`): charts of the published runs, FP16 vs FP8
  KV cache, and the vLLM planner in the browser, with a predicted-vs-actual KV check (−9.5% FP16, +5.1% FP8).
- 2026-10-06: planner (vLLM KV blocks, llama.cpp `-ngl` from GGUF tensor sizes), streaming benchmark with NVML
  and Prometheus metrics, MCP server (FastMCP), docker-compose profiles, run matrix, CI.
