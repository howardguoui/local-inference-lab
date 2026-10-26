"""A stand-in for vLLM: OpenAI-compatible streaming chat, /v1/models and Prometheus /metrics.

Each request waits a fixed prefill time, then streams tokens at a fixed rate. The KV
cache gauge rises with the number of active requests, and the preemption counter
ticks when more than CAPACITY requests run at once, like a real server whose cache
is full. A system prompt seen before counts as a prefix cache hit (and prefills faster),
like vLLM's automatic prefix caching.

Non-streaming requests (the accuracy check) get `reply(question)` back, cut to max_tokens words
with finish_reason "length" like a real server; a reply of None is a server error (HTTP 500).
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from collections.abc import Callable

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

CAPACITY = 4


def build_app(
    prefill_s: float = 0.03,
    token_s: float = 0.004,
    tokens: int = 20,
    reply: Callable[[str], str | None] | None = None,
) -> FastAPI:
    app = FastAPI()
    state = {"active": 0, "peak": 0, "preemptions": 0, "prefix_hits": 0, "prefix_queries": 0}
    seen_systems: set[str] = set()

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": "fake/qwen-7b", "object": "model"}]}

    @app.get("/metrics")
    def metrics():
        usage = min(1.0, state["active"] / CAPACITY)
        body = "\n".join(
            [
                "# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.",
                "# TYPE vllm:kv_cache_usage_perc gauge",
                f'vllm:kv_cache_usage_perc{{engine="0",model_name="fake/qwen-7b"}} {usage}',
                f'vllm:num_requests_running{{engine="0"}} {min(state["active"], CAPACITY)}',
                f'vllm:num_requests_waiting{{engine="0"}} {max(0, state["active"] - CAPACITY)}',
                f'vllm:num_preemptions_total{{engine="0"}} {state["preemptions"]}',
                f'vllm:prefix_cache_hits_total{{engine="0"}} {state["prefix_hits"]}',
                f'vllm:prefix_cache_queries_total{{engine="0"}} {state["prefix_queries"]}',
                'vllm:cache_config_info{block_size="16",cache_dtype="auto",enable_prefix_caching="True",'
                'gpu_memory_utilization="0.9",num_gpu_blocks="9011"} 1.0',
            ]
        )
        return PlainTextResponse(body + "\n")

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        if body.get("model") != "fake/qwen-7b":
            return JSONResponse({"error": "unknown model"}, status_code=404)
        n = min(tokens, body.get("max_tokens", tokens))
        if not body.get("ignore_eos"):
            n = min(n, 5)  # the "model" reaches end-of-sequence early unless told to ignore it
        prompt = body["messages"][-1]["content"]
        if not body.get("stream"):
            text = reply(prompt) if reply else "t0 t1 t2"
            if text is None:
                return JSONResponse({"error": {"message": "internal error", "code": 500}}, status_code=500)
            words = text.split(" ")
            limit = body.get("max_tokens", len(words))
            finish = "length" if len(words) > limit else "stop"
            message = {"role": "assistant", "content": " ".join(words[:limit])}
            usage = {"prompt_tokens": 60, "completion_tokens": min(limit, len(words)), "total_tokens": 0}
            return {"choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage}
        system = next((m["content"] for m in body["messages"] if m["role"] == "system"), None)
        hit = system is not None and system in seen_systems
        if system is not None:
            seen_systems.add(system)
        state["prefix_queries"] += 600
        state["prefix_hits"] += 480 if hit else 0  # the system prompt's share of the 600 prompt tokens

        async def gen():
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            if state["active"] > CAPACITY:
                state["preemptions"] += 1
            try:
                await asyncio.sleep(prefill_s / 5 if hit else prefill_s)
                if "FAIL" in prompt:  # an error reported inside a 200 stream, as vLLM and llama.cpp do
                    yield f"data: {json.dumps({'error': {'message': 'context length exceeded', 'code': 400}})}\n\n"
                    return
                for i in range(n):
                    finish = ("length" if body.get("ignore_eos") else "stop") if i == n - 1 else None
                    chunk = {"choices": [{"index": 0, "delta": {"content": f"t{i} "}, "finish_reason": finish}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                    await asyncio.sleep(token_s)
                usage = {"prompt_tokens": 600, "completion_tokens": n, "total_tokens": 600 + n}
                yield f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                state["active"] -= 1

        return StreamingResponse(gen(), media_type="text/event-stream")

    app.state.counters = state
    return app


class ServerThread:
    """Run the fake server on a free localhost port for the duration of a test."""

    def __init__(self, app: FastAPI):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> ServerThread:
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        return self

    def __exit__(self, *exc) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)
