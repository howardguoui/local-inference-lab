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
- [ ] **Run the accuracy check** on the GPU (`run_matrix.sh` now runs `inference-lab accuracy` per config) and
      put each config's score next to its throughput in the README.
- [ ] **Speculative decoding:** vLLM n-gram / draft-model speculation on the chat scenario; acceptance rate and
      tokens/s change.
- [ ] **SGLang backend** as a fourth OpenAI-compatible server profile.
- [ ] **Measure power efficiency** on the GPU (every `inference-lab bench` level now records energy) and put
      each config's tokens per joule next to its throughput in the README.

## Shipped

- 2026-10-30: MCP `compare_configs` tool: "which config is fastest under N streams" from the saved runs (latest per
  config). It ranks each config's measured level at N streams by throughput (default), TTFT p50/p95, TPOT p50,
  end-to-end p95 or tokens per joule, separately per prompt size, with failed-request levels last and a note where
  generation stopped early; configs without a level at N are listed with the levels they did run (no
  interpolation). Unmeasured N, unknown prompt sizes, missing energy data and an empty `results/` come back as
  tool errors that say what was measured; the schema bounds `streams` and enumerates the metrics.
- 2026-10-30: power efficiency: the NVML sampler records the energy the GPU board used during each level (the
  driver's total energy counter where the GPU has one, else the power samples integrated with the trapezoid rule)
  and its mean power; every level saves `mean_power_w`, `energy_j`, `energy_source` and `tokens_per_joule`
  (output tokens / energy), and `results/latest.md` gets a "Power efficiency" table for runs that have them. The
  published 2026-10-06/07 runs saved only `peak_power_w` per level, which can't give energy, so latest.md has no
  such table yet; numbers come with the next GPU run (item above).
- 2026-10-26: quality check per config: `inference-lab accuracy` scores a server on 50 arithmetic word problems
  generated from templates with a fixed seed (answers computed in code, set fingerprinted so runs stay
  comparable), with greedy decoding and exact-match scoring of the final number. Each run is saved as
  `results/accuracy-<label>-*.json`; `results/latest.md` gets an "Answer accuracy" table (correct, truncated,
  unparsed, errors) and `run_matrix.sh` runs it for every config. Scores come with the next GPU run (item above).
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
