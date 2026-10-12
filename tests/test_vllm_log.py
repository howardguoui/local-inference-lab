"""vLLM startup log parsing and plan calibration.

The log below is synthetic: each line is one of vLLM's format strings (vllm/v1/worker/gpu_worker.py,
gpu_model_runner.py, v1/core/kv_cache_utils.py, utils/mem_utils.py) filled with made-up numbers, wrapped in the
prefixes and colours a containerised V1 engine prints.
"""

import pytest

from inference_lab.cli import main
from inference_lab.models import PRESETS
from inference_lab.planner import GIB, calibrate_vllm, kv_bytes_per_token, plan_vllm
from inference_lab.vllm_log import parse_vllm_log, read_vllm_log

LOG = """\
INFO 10-07 13:00:01 [__init__.py:216] Automatically detected platform cuda.
(EngineCore_DP0 pid=212) INFO 10-07 13:00:31 [gpu_model_runner.py:2602] Model loading took 5.25 GiB memory \
and 21.402816 seconds
(EngineCore_DP0 pid=212) DEBUG 10-07 13:00:40 [gpu_worker.py:301] Memory profiling takes 4.12 seconds. Total \
non KV cache memory: 6.31GiB; torch peak memory increase: 0.98GiB; total consumed (from mem_get_info): \
6.05GiB; weights memory: 5.25GiB.
(EngineCore_DP0 pid=212) INFO 10-07 13:00:40 [gpu_model_runner.py:3880] Estimated CUDA graph memory: 0.52 GiB total
(EngineCore_DP0 pid=212) \x1b[1;36mINFO\x1b[0m 10-07 13:00:40 [gpu_worker.py:298] Available KV cache memory: \
7.20 GiB
(EngineCore_DP0 pid=212) INFO 10-07 13:00:41 [kv_cache_utils.py:1087] GPU KV cache size: 134,816 tokens
(EngineCore_DP0 pid=212) INFO 10-07 13:00:41 [kv_cache_utils.py:1091] Maximum concurrency for 8,192 tokens per \
request: 16.46x
(EngineCore_DP0 pid=212) INFO 10-07 13:00:55 [gpu_model_runner.py:3480] Graph capturing finished in 14 secs, \
took 0.49 GiB
(EngineCore_DP0 pid=212) INFO 10-07 13:00:55 [gpu_worker.py:960] CUDA graph pool memory: 0.49 GiB (actual), \
0.52 GiB (estimated), difference: 0.03 GiB (6.1%).
(APIServer pid=1) INFO 10-07 13:00:58 [api_server.py:1971] Starting vLLM API server 0 on http://0.0.0.0:8000
"""


def test_parses_every_memory_figure_through_prefixes_and_colours():
    s = parse_vllm_log(LOG)
    assert s.kv_available_gib == 7.20
    assert s.kv_cache_tokens == 134_816
    assert s.max_model_len == 8192
    assert s.max_concurrency == 16.46
    assert s.weights_gib == 5.25
    assert s.non_kv_gib == 6.31
    assert s.cuda_graph_estimate_gib == 0.52
    assert s.cuda_graph_gib == 0.49


def test_last_startup_wins_and_optional_lines_may_be_missing():
    restart = "INFO Available KV cache memory: 3.2 GiB\nINFO GPU KV cache size: 59,920 tokens\n"
    s = parse_vllm_log(LOG + restart)
    assert (s.kv_available_gib, s.kv_cache_tokens) == (3.2, 59_920)
    bare = parse_vllm_log("Available KV cache memory: 1.0 GiB")
    assert bare.kv_cache_tokens is None and bare.weights_gib is None and bare.cuda_graph_gib is None


def test_log_without_kv_line_is_an_error_that_says_what_to_pass():
    with pytest.raises(ValueError, match="Available KV cache memory.*docker compose logs"):
        parse_vllm_log("ERROR torch.OutOfMemoryError: CUDA out of memory.")


def test_fitted_overhead_reproduces_the_logged_kv_memory():
    model = PRESETS["qwen2.5-7b"]
    plan = plan_vllm(model, gpu_gib=15.92, weights_gib=5.19, max_model_len=8192)
    cal = calibrate_vllm(plan, parse_vllm_log(LOG))
    # 15.92 x 0.9 - 5.19 - 7.20 = 1.938
    assert cal.fitted_overhead_gib == pytest.approx(1.94)
    assert cal.actual_kv_tokens == 134_816 and cal.actual_source == "log"
    assert cal.error_pct == pytest.approx(100 * (plan.kv_tokens - 134_816) / 134_816, abs=0.05)
    # planning again with the fitted overhead lands on the logged KV memory, to vLLM's two-decimal rounding
    refit = plan_vllm(model, 15.92, 5.19, 8192, overhead_gib=cal.fitted_overhead_gib)
    assert refit.kv_budget_gib == pytest.approx(7.20, abs=0.01)
    assert abs(refit.kv_tokens - 134_816) / 134_816 < 0.003
    assert "% over what vLLM allocated" in cal.advice and "--overhead-gib 1.94" in cal.advice


def test_tokens_fall_back_to_kv_memory_and_weight_mismatch_is_reported():
    model = PRESETS["qwen2.5-7b"]
    plan = plan_vllm(model, gpu_gib=15.92, weights_gib=5.19, max_model_len=8192, overhead_gib=1.94)
    startup = parse_vllm_log("Model loading took 5.6 GiB memory\nAvailable KV cache memory: 7.20 GiB")
    cal = calibrate_vllm(plan, startup)
    per_block = kv_bytes_per_token(model, 2) * 16
    assert cal.actual_source == "kv_memory"
    assert cal.actual_kv_tokens == int(7.20 * GIB // per_block) * 16
    assert abs(cal.error_pct) <= 2 and cal.advice.startswith("The plan is within")
    assert "loaded 5.60 GiB of weights" in cal.advice


def test_cli_plan_vllm_takes_max_model_len_from_the_log(tmp_path, capsys):
    log = tmp_path / "vllm.log"
    log.write_text(LOG.replace("8,192 tokens per", "16,384 tokens per"), encoding="utf-8")
    assert read_vllm_log(log).max_model_len == 16_384
    main(
        ["plan", "vllm", "--model", "qwen2.5-7b", "--weights-gib", "5.19", "--gpu-gib", "15.92", "--vllm-log", str(log)]
    )
    out = capsys.readouterr().out
    assert "max_model_len              16,384" in out
    assert "fitted_overhead_gib" in out and "1.94" in out
    assert "actual_kv_tokens     134,816" in out


def test_cli_rejects_a_log_from_a_server_that_never_started(tmp_path):
    log = tmp_path / "crash.log"
    log.write_text("ValueError: not enough KV cache memory", encoding="utf-8")
    with pytest.raises(SystemExit, match="crash.log: No 'Available KV cache memory'"):
        main(
            [
                "plan",
                "vllm",
                "--model",
                "qwen2.5-7b",
                "--weights-gib",
                "5.19",
                "--gpu-gib",
                "16",
                "--vllm-log",
                str(log),
            ]
        )
