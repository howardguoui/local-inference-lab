# Roadmap

Shipped items move to the bottom with the date. Results are only ever committed from real runs on the GPU
machine (`scripts/publish_results.sh`); nothing in `results/` is hand-written.

## Next

- [ ] **First full benchmark run** on the RTX 5070 Ti: `scripts/run_matrix.sh` (with `OFFLOAD=1`), publish
      `results/latest.md`, and summarize the findings at the top of the README.
- [ ] **Planner calibration:** parse vLLM's startup log ("Available KV cache memory", CUDA graph memory) and
      report predicted vs actual KV blocks with the error in percent; tune `overhead_gib` defaults from real runs.
- [ ] **Prefix caching scenario:** shared 2k-token system prompt plus unique questions; record vLLM's prefix cache
      hit rate and the TTFT drop against unique prompts.
- [ ] **Quality check per config:** a small fixed answer-accuracy set (e.g. 50 arithmetic word problems with exact
      answers) so FP8 KV cache and Q4 quantization show any accuracy cost next to their speed gain.
- [ ] **Speculative decoding:** vLLM n-gram / draft-model speculation on the chat scenario; acceptance rate and
      tokens/s change.
- [ ] **SGLang backend** as a fourth OpenAI-compatible server profile.
- [ ] **Power efficiency:** tokens per joule per config from the NVML power samples.
- [ ] **MCP:** a `compare_configs` tool that answers "which config is fastest under N streams" from saved runs.

## Shipped

- 2026-10-09: charts of throughput and TTFT p95 against concurrency per server config, written as SVG from
  `results/*.json` by `inference-lab report` (standard library instead of matplotlib: no new dependency).
- 2026-10-07: demo page on GitHub Pages (`inference-lab demo` → `docs/`): charts of the published runs, FP16 vs FP8
  KV cache, and the vLLM planner in the browser, with a predicted-vs-actual KV check (−9.5% FP16, +5.1% FP8).
- 2026-10-06: planner (vLLM KV blocks, llama.cpp `-ngl` from GGUF tensor sizes), streaming benchmark with NVML
  and Prometheus metrics, MCP server (FastMCP), docker-compose profiles, run matrix, CI.
