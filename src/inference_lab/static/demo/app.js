(function () {
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmt = (x, d = 0) => (x == null ? '–' : Number(x).toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: d }));
  const CONFIGS = [
    ['vllm-fp8kv', 'vLLM, FP8 KV cache', '--s1'],
    ['vllm-fp16kv', 'vLLM, FP16 KV cache', '--s2'],
    ['llamacpp-f16kv', 'llama.cpp, F16 KV cache', '--s3'],
    ['llamacpp-q8kv', 'llama.cpp, Q8 KV cache', '--s4'],
    ['ollama-q4km', 'Ollama', '--s5'],
  ];
  const configOf = (label) => CONFIGS.find(([k]) => label === k || label.startsWith(k + '-'));
  const color = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

  // ---- charts ----
  function lineChart(svg, xs, series, opts) {
    const W = 440, H = 260, L = 50, R = 12, T = 12, B = 34;
    const ys = series.flatMap((s) => s.points.map((p) => p.y)).filter((y) => y > 0);
    let y0, y1, ticks, Y;
    if (opts.log) {
      y0 = Math.pow(10, Math.floor(Math.log10(Math.min(...ys))));
      y1 = Math.pow(10, Math.ceil(Math.log10(Math.max(...ys))));
      ticks = []; for (let t = y0; t <= y1 * 1.0001; t *= 10) ticks.push(t);
      Y = (y) => T + (H - T - B) * (1 - (Math.log10(y) - Math.log10(y0)) / (Math.log10(y1) - Math.log10(y0)));
    } else {
      const max = Math.max(...ys), step = Math.pow(10, Math.floor(Math.log10(max)));
      const unit = max / step > 5 ? step : max / step > 2 ? step / 2 : step / 4;
      y0 = 0; y1 = Math.ceil(max / unit) * unit;
      ticks = []; for (let t = 0; t <= y1 + 1e-9; t += unit) ticks.push(t);
      Y = (y) => T + (H - T - B) * (1 - y / y1);
    }
    const X = (i) => L + (xs.length === 1 ? (W - L - R) / 2 : (i * (W - L - R)) / (xs.length - 1));
    let out = '';
    ticks.forEach((t) => {
      out += `<line x1="${L}" x2="${W - R}" y1="${Y(t)}" y2="${Y(t)}" stroke="${color('--grid')}"/>`;
      out += `<text x="${L - 6}" y="${Y(t) + 4}" text-anchor="end">${opts.tick(t)}</text>`;
    });
    xs.forEach((x, i) => { out += `<text x="${X(i)}" y="${H - B + 16}" text-anchor="middle">${x}</text>`; });
    out += `<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">concurrent users</text>`;
    series.forEach((s) => {
      const pts = s.points.map((p) => [X(xs.indexOf(p.x)), Y(p.y)]);
      out += `<polyline fill="none" stroke="${color(s.color)}" stroke-width="2.5" points="${pts.map((p) => p.join(',')).join(' ')}"><title>${esc(s.name)}</title></polyline>`;
      s.points.forEach((p, i) => {
        out += `<circle cx="${pts[i][0]}" cy="${pts[i][1]}" r="3.5" fill="${color(s.color)}"><title>${esc(s.name)}: ${opts.tip(p.y)} at ${p.x} users</title></circle>`;
      });
    });
    svg.innerHTML = out;
  }

  function renderServing(data, scenario) {
    const runs = data.runs.filter((r) => r.scenario === scenario && configOf(r.label));
    runs.sort((a, b) => CONFIGS.indexOf(configOf(a.label)) - CONFIGS.indexOf(configOf(b.label)));
    const xs = [...new Set(runs.flatMap((r) => r.levels.map((l) => l.concurrency)))].sort((a, b) => a - b);
    const series = (key) => runs.map((r) => ({ name: configOf(r.label)[1], color: configOf(r.label)[2],
      points: r.levels.map((l) => ({ x: l.concurrency, y: l[key] })) }));
    lineChart($('#c-tput'), xs, series('throughput_tok_s'), { tick: (t) => fmt(t), tip: (y) => `${fmt(y, 1)} tok/s` });
    const ttft = series('ttft_p50_ms').map((s) => ({ ...s, points: s.points.map((p) => ({ x: p.x, y: p.y / 1000 })) }));
    lineChart($('#c-ttft'), xs, ttft, { log: true, tick: (t) => (t < 1 ? t.toFixed(1) : fmt(t)), tip: (y) => `${fmt(y, 2)} s` });
    $('#legend').innerHTML = runs.map((r) => `<li><i style="background:${color(configOf(r.label)[2])}"></i>${esc(configOf(r.label)[1])}</li>`).join('');
    const rows = runs.flatMap((r) => r.levels.map((l) => `<tr><td>${esc(configOf(r.label)[1])}</td><td class="num">${l.concurrency}</td>
      <td class="num">${fmt(l.throughput_tok_s, 1)}</td><td class="num">${fmt(l.ttft_p50_ms / 1000, 2)} s</td>
      <td class="num">${fmt(l.ttft_p95_ms / 1000, 2)} s</td><td class="num">${fmt(l.tpot_p50_ms, 1)} ms</td>
      <td class="num">${fmt(l.peak_vram_gib, 2)}</td><td class="num">${l.errors}/${l.requests}</td></tr>`));
    $('#t-runs').innerHTML = `<thead><tr><th>Config</th><th class="num">Users</th><th class="num">Tokens/s</th>
      <th class="num">First token p50</th><th class="num">p95</th><th class="num">Per token</th><th class="num">Peak VRAM (GiB)</th>
      <th class="num">Errors</th></tr></thead><tbody>${rows.join('')}</tbody>`;
  }

  function level(data, label, conc) {
    const r = data.runs.find((x) => x.label === label);
    return r && r.levels.find((l) => l.concurrency === conc);
  }

  function renderFacts(data) {
    const v = level(data, 'vllm-fp8kv-chat', 16), l = level(data, 'llamacpp-f16kv-chat', 16), o = level(data, 'ollama-q4km', 16);
    const cal = Object.fromEntries(data.calibration.map((c) => [c.kv_cache_dtype, c]));
    const tiles = [];
    if (v && l && o) {
      tiles.push([`${fmt(v.throughput_tok_s)} tok/s`, `vLLM at 16 users, vs ${fmt(l.throughput_tok_s)} for llama.cpp and ${fmt(o.throughput_tok_s)} for Ollama`]);
      tiles.push([`${fmt(v.ttft_p50_ms / 1000, 1)} s vs ${fmt(o.ttft_p50_ms / 1000)} s`, 'median wait for the first token at 16 users: vLLM vs Ollama']);
    }
    if (cal.auto && cal.fp8) tiles.push([`${fmt(cal.fp8.actual_tokens / cal.auto.actual_tokens, 1)}× the KV cache`, `FP8 vs FP16: ${fmt(cal.fp8.actual_tokens)} vs ${fmt(cal.auto.actual_tokens)} tokens`]);
    if (data.calibration.length) {
      const worst = Math.max(...data.calibration.map((c) => Math.abs(c.error_pct)));
      tiles.push([`within ${fmt(worst, 0)}%`, 'planner prediction vs the KV cache vLLM really allocated']);
    }
    $('#facts').innerHTML = tiles.map(([b, s]) => `<div class="fact"><b>${b}</b><span>${s}</span></div>`).join('');
    $('#when').textContent = `Measured on an ${data.gpu} (${fmt(data.gpu_gib, 2)} GiB), ${data.run_dates[0]} to ${data.run_dates[data.run_dates.length - 1]}.`;
  }

  function renderKv(data) {
    const cal = data.calibration;
    const max = Math.max(...cal.map((c) => c.actual_tokens));
    $('#kv-bars').innerHTML = cal.map((c) => `<div><span>${c.kv_cache_dtype === 'fp8' ? 'FP8' : 'FP16'} KV cache</span>
      <div class="bar" style="width:${Math.max(30, (100 * c.actual_tokens) / max)}%;background:${color(c.kv_cache_dtype === 'fp8' ? '--s1' : '--s2')}">${fmt(c.actual_tokens)} tokens</div></div>`).join('');
    const concs = [8, 16, 32, 48];
    const rows = concs.map((n) => {
      const a = level(data, 'vllm-fp16kv-long', n), b = level(data, 'vllm-fp8kv-long', n);
      if (!a || !b) return '';
      const pct = (x) => (x == null ? '–' : `${Math.round(x * 100)}%`);
      return `<tr><td class="num">${n}</td><td class="num">${pct(a.peak_kv_cache_usage)} / ${pct(b.peak_kv_cache_usage)}</td>
        <td class="num">${fmt(a.preemptions)} / ${fmt(b.preemptions)}</td><td class="num">${fmt(a.throughput_tok_s)} / ${fmt(b.throughput_tok_s)}</td>
        <td class="num">${fmt(a.ttft_p95_ms / 1000, 1)} / ${fmt(b.ttft_p95_ms / 1000, 1)} s</td></tr>`;
    }).join('');
    $('#t-kv').innerHTML = `<thead><tr><th class="num">Users (4k prompts)</th><th class="num">Peak KV use, FP16 / FP8</th>
      <th class="num">Preemptions</th><th class="num">Tokens/s</th><th class="num">First token p95</th></tr></thead><tbody>${rows}</tbody>`;
  }

  // ---- planner ----
  function initPlanner(data) {
    const spec = data.planner, cal = data.calibration;
    const sel = $('#p-model');
    sel.innerHTML = Object.keys(spec.models).map((m) => `<option>${esc(m)}</option>`).join('');
    const ref = cal[0];
    sel.value = ref ? ref.preset : Object.keys(spec.models)[0];
    $('#p-weights').value = ref ? ref.weights_gib : 5;
    $('#p-gpu').value = data.gpu_gib;
    $('#p-util').value = spec.gpu_memory_utilization;

    function hints() {
      const m = spec.models[sel.value], gib = (bytes) => ((m.params_b * 1e9 * bytes) / spec.gib).toFixed(2);
      const opts = [['FP16', gib(2), '2 bytes per parameter'], ['8-bit', gib(1), '1 byte per parameter'], ['4-bit', gib(0.5), 'half a byte per parameter, before quantization scales']];
      if (ref && sel.value === ref.preset) opts.unshift(['AWQ checkpoint we ran', ref.weights_gib.toFixed(2), 'sum of the weight files']);
      $('#p-hints').innerHTML = opts.map(([n, v, t]) => `<button type="button" data-w="${v}" title="${t}">${n} ≈ ${v}</button>`).join('');
      $('#p-hints').querySelectorAll('button').forEach((b) => b.addEventListener('click', () => { $('#p-weights').value = b.dataset.w; update(); }));
    }

    function update() {
      const kv = document.querySelector('input[name=p-kv]:checked').value;
      const opts = { model: sel.value, gpu_gib: +$('#p-gpu').value, weights_gib: +$('#p-weights').value,
        max_model_len: +$('#p-len').value, gpu_memory_utilization: +$('#p-util').value, kv_cache_dtype: kv };
      $('#p-util-v').textContent = `${Math.round(opts.gpu_memory_utilization * 100)}%`;
      const p = Planner.planVllm(spec, opts);
      const verdict = p.fits
        ? `<p class="verdict ok">Fits: ${fmt(p.max_concurrent_at_max_len)} requests at the full ${fmt(opts.max_model_len)}-token context can be in memory at once.</p>`
        : `<p class="verdict no">vLLM would refuse to start: one ${fmt(opts.max_model_len)}-token request needs ${p.gib_per_sequence.toFixed(2)} GiB of KV cache but only ${p.kv_budget_gib.toFixed(2)} GiB is left. Try FP8, a shorter context, or a smaller model.</p>`;
      const match = cal.find((c) => c.preset === opts.model && c.kv_cache_dtype === kv && Math.abs(c.weights_gib - opts.weights_gib) < 0.01
        && Math.abs(c.gpu_gib - opts.gpu_gib) < 0.01 && Math.abs(c.gpu_memory_utilization - opts.gpu_memory_utilization) < 0.001);
      const check = match
        ? `<p class="check">This is the setup we benchmarked: vLLM actually allocated ${fmt(match.actual_tokens)} tokens, so this plan is off by ${match.error_pct > 0 ? '+' : ''}${fmt(match.error_pct, 1)}%.</p>`
        : '';
      $('#p-out').innerHTML = `<div class="big">${fmt(p.kv_tokens)} tokens</div><div>of KV cache (${fmt(p.kv_blocks)} blocks of ${spec.block_size})</div>
        <dl><dt>Memory left for KV cache</dt><dd>${p.kv_budget_gib.toFixed(2)} GiB</dd>
        <dt>KV cache per token</dt><dd>${fmt(p.kv_bytes_per_token / 1024)} KiB</dd>
        <dt>Activations + CUDA graphs</dt><dd>${spec.overhead_gib} GiB (estimate)</dd></dl>${verdict}${check}`;
    }

    sel.addEventListener('change', () => { hints(); update(); });
    $('#pform').addEventListener('input', update);
    hints(); update();
  }

  fetch('data.json').then((r) => r.json()).then((data) => {
    renderFacts(data);
    let scenario = 'chat';
    document.querySelectorAll('[data-scenario]').forEach((b) => b.addEventListener('click', () => {
      scenario = b.dataset.scenario;
      document.querySelectorAll('[data-scenario]').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
      $('#ttft-cap').textContent = `Median time to first token (seconds, log scale, lower is better)`;
      renderServing(data, scenario);
    }));
    renderServing(data, scenario);
    renderKv(data);
    initPlanner(data);
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { renderServing(data, scenario); renderKv(data); });
  }).catch((e) => { $('#when').textContent = `Could not load data.json (${e.message}).`; });
})();
