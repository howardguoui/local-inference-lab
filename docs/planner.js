// vLLM KV cache planner: the same arithmetic as plan_vllm in planner.py, using the constants and model
// presets that `inference-lab demo` exports to data.json. tests/test_demo.py runs this file under Node and
// checks it against plan_vllm on a grid of inputs.
(function (root) {
  function kvBytesPerToken(model, bytesPerValue) {
    return model.n_layers * model.n_kv_heads * model.head_dim * (bytesPerValue + bytesPerValue);
  }

  function planVllm(spec, opts) {
    const model = spec.models[opts.model];
    const util = opts.gpu_memory_utilization ?? spec.gpu_memory_utilization;
    const dtype = opts.kv_cache_dtype ?? 'auto';
    const overhead = opts.overhead_gib ?? spec.overhead_gib;
    const blockSize = spec.block_size;
    const perToken = kvBytesPerToken(model, spec.kv_bytes[dtype]);
    const budget = opts.gpu_gib * util - opts.weights_gib - overhead;
    const blocks = Math.max(0, Math.floor((budget * spec.gib) / (perToken * blockSize)));
    const tokens = blocks * blockSize;
    return {
      kv_bytes_per_token: perToken,
      kv_budget_gib: Math.round(Math.max(budget, 0) * 100) / 100,
      kv_blocks: blocks,
      kv_tokens: tokens,
      max_concurrent_at_max_len: opts.max_model_len ? Math.floor(tokens / opts.max_model_len) : 0,
      fits: tokens >= opts.max_model_len,
      gib_per_sequence: (opts.max_model_len * perToken) / spec.gib,
    };
  }

  const api = { kvBytesPerToken, planVllm };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Planner = api;
})(typeof window !== 'undefined' ? window : globalThis);
