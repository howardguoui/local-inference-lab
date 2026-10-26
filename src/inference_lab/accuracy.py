"""Answer accuracy per server config: a fixed set of arithmetic word problems with exact answers.

Speed numbers alone hide what a cheaper config may cost in quality: an FP8 KV cache, a q8_0 cache or a
different 4-bit format can make a model slightly worse at multi-step reasoning. This runs the same problems
against any OpenAI-compatible server and counts exact matches, so each config's accuracy sits next to its
throughput in results/latest.md.

The problems are generated from templates with a fixed seed and their answers are computed here, not written
by hand, so the set is the same on every machine. `fingerprint` hashes the questions and answers: two runs are
comparable only when their fingerprints match. Requests use greedy decoding (temperature 0) and a fixed seed;
batching can still change floating-point results slightly, so treat differences of one or two problems as noise.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

import httpx

from .backends import Backend, discover_model
from .gpu import gpu_snapshot

PROBLEM_SET = "arith-v1"
SYSTEM_PROMPT = (
    "Solve the math word problem. Reason briefly, then write the final answer on its own line as "
    "'Answer: <number>', digits only, without units."
)


@dataclass(frozen=True)
class Problem:
    id: int
    question: str
    answer: int


def _shop(r: random.Random) -> tuple[str, int]:
    a, b, c, d = r.randint(2, 9), r.randint(1, 4), r.randint(2, 6), r.randint(2, 8)
    total = a * c + b * d
    bill = (total // 10 + 1) * 10 + r.choice([0, 10, 40])
    q = (
        f"A shop sells notebooks for ${a} each and pens for ${b} each. Mia buys {c} notebooks and {d} pens "
        f"and pays with ${bill}. How many dollars of change does she get?"
    )
    return q, bill - total


def _printer(r: random.Random) -> tuple[str, int]:
    p, h, m = r.randint(12, 45), r.randint(1, 3), r.randint(5, 55)
    q = f"A printer prints {p} pages per minute. How many pages does it print in {h} hours and {m} minutes?"
    return q, p * (60 * h + m)


def _cookies(r: random.Random) -> tuple[str, int]:
    trays, per_tray, friends = r.randint(3, 8), r.randint(10, 24), r.randint(3, 7)
    total = trays * per_tray
    kept = total % friends + friends * r.randint(0, 2)
    q = (
        f"A baker makes {trays} trays of {per_tray} cookies. She keeps {kept} cookies and shares the rest "
        f"equally among {friends} friends. How many cookies does each friend get?"
    )
    return q, (total - kept) // friends


def _train(r: random.Random) -> tuple[str, int]:
    v, h1, w, h2 = r.randint(60, 120), r.randint(2, 5), r.randint(40, 90), r.randint(1, 4)
    q = (
        f"A train travels at {v} km/h for {h1} hours, then at {w} km/h for {h2} hours. "
        "How many kilometres does it travel in total?"
    )
    return q, v * h1 + w * h2


def _ages(r: random.Random) -> tuple[str, int]:
    sister = r.randint(4, 12)
    gap, factor, years = r.randint(2, 6), r.randint(3, 5), r.randint(2, 9)
    q = (
        f"Tom is {sister + gap} years old and his sister is {gap} years younger. Their father is {factor} "
        f"times as old as the sister. How old will the father be in {years} years?"
    )
    return q, sister * factor + years


def _discount(r: random.Random) -> tuple[str, int]:
    price, off, qty = 20 * r.randint(2, 15), r.choice([10, 20, 25, 50]), r.randint(2, 4)
    q = (
        f"A jacket costs ${price}. During a sale every jacket is {off} percent off. "
        f"How many dollars do {qty} jackets cost during the sale?"
    )
    return q, qty * price * (100 - off) // 100


def _savings(r: random.Random) -> tuple[str, int]:
    weekly, weeks = r.randint(8, 30), r.randint(6, 20)
    spent = r.randint(weekly * weeks // 3, weekly * weeks - 1)
    q = (
        f"Sam saves ${weekly} every week for {weeks} weeks, then spends ${spent} on a bike. "
        "How many dollars does he have left?"
    )
    return q, weekly * weeks - spent


def _tank(r: random.Random) -> tuple[str, int]:
    rate_in, rate_out = r.randint(9, 20), r.randint(2, 8)
    minutes, start = r.randint(10, 40), r.randint(20, 200)
    q = (
        f"A tank holds {start} litres of water. A pipe adds {rate_in} litres per minute while a leak loses "
        f"{rate_out} litres per minute. How many litres are in the tank after {minutes} minutes?"
    )
    return q, start + (rate_in - rate_out) * minutes


TEMPLATES: list[Callable[[random.Random], tuple[str, int]]] = [
    _shop,
    _printer,
    _cookies,
    _train,
    _ages,
    _discount,
    _savings,
    _tank,
]


def make_problems(n: int = 50, seed: int = 0) -> list[Problem]:
    """n problems cycling through the templates, with numbers drawn from a seeded RNG."""
    r = random.Random(seed)
    out = []
    for i in range(n):
        question, answer = TEMPLATES[i % len(TEMPLATES)](r)
        out.append(Problem(i + 1, question, answer))
    return out


def fingerprint(problems: list[Problem]) -> str:
    h = hashlib.sha256()
    for p in problems:
        h.update(f"{p.id}\t{p.question}\t{p.answer}\n".encode())
    return h.hexdigest()[:12]


_NUM = r"-?\$?\s*\d[\d,]*(?:\.\d+)?"
_PATTERNS = [
    re.compile(r"\\boxed\{\s*(" + _NUM + r")\s*\}"),
    re.compile(r"answer\s*(?:is)?\s*[:=]?\s*\**\s*(" + _NUM + ")", re.IGNORECASE),
]
_ANY_NUM = re.compile(_NUM)


def _to_number(s: str) -> float | None:
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return None


def extract_answer(text: str) -> float | None:
    """The model's final answer: the last \\boxed{...}, else the last 'Answer: N', else the last number."""
    for pattern in _PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return _to_number(matches[-1])
    matches = _ANY_NUM.findall(text)
    return _to_number(matches[-1]) if matches else None


@dataclass
class ItemResult:
    id: int
    expected: int
    predicted: float | None = None
    correct: bool = False
    finish_reason: str | None = None
    output_tokens: int | None = None
    error: str | None = None
    tail: str = ""  # end of the reply, enough to see why a wrong answer was wrong


@dataclass
class AccuracyResult:
    label: str
    backend: dict
    model: str
    started_at: str
    problem_set: str
    seed: int
    n: int
    fingerprint: str
    max_tokens: int
    correct: int
    errors: int
    unparsed: int
    truncated: int  # replies cut off at max_tokens: raise --max-tokens if this is not 0
    accuracy: float
    wall_s: float
    gpu: dict | None = None
    kind: str = "accuracy"
    items: list[ItemResult] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


async def ask(client: httpx.AsyncClient, model: str, problem: Problem, max_tokens: int, seed: int) -> ItemResult:
    item = ItemResult(problem.id, problem.answer)
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": problem.question}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": seed,
    }
    try:
        resp = await client.post("/chat/completions", json=payload, timeout=300)
        if resp.status_code != 200:
            item.error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            return item
        body = resp.json()
        if body.get("error"):
            item.error = str(body["error"])[:200]
            return item
        choice = body["choices"][0]
        text = choice["message"].get("content") or ""
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
        item.error = f"{type(exc).__name__}: {exc}"[:200]
        return item
    item.finish_reason = choice.get("finish_reason")
    item.output_tokens = (body.get("usage") or {}).get("completion_tokens")
    item.tail = text[-200:]
    item.predicted = extract_answer(text)
    item.correct = item.predicted is not None and item.predicted == problem.answer
    return item


async def run_accuracy(
    backend: Backend,
    n: int = 50,
    concurrency: int = 8,
    max_tokens: int = 512,
    label: str | None = None,
    seed: int = 0,
    transport: httpx.AsyncBaseTransport | None = None,
    nvml=None,
) -> AccuracyResult:
    if n < 1 or concurrency < 1:
        raise ValueError("--problems and --concurrency must be at least 1")
    problems = make_problems(n, seed)
    gate = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=backend.base_url, transport=transport) as client:
        model = backend.model or await discover_model(client)
        started = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        t0 = time.perf_counter()

        async def one(p: Problem) -> ItemResult:
            async with gate:
                return await ask(client, model, p, max_tokens, seed)

        items = await asyncio.gather(*(one(p) for p in problems))
        wall = time.perf_counter() - t0
    correct = sum(i.correct for i in items)
    return AccuracyResult(
        label=label or backend.name,
        backend=backend.as_dict(),
        model=model,
        started_at=started,
        problem_set=PROBLEM_SET,
        seed=seed,
        n=n,
        fingerprint=fingerprint(problems),
        max_tokens=max_tokens,
        correct=correct,
        errors=sum(i.error is not None for i in items),
        unparsed=sum(i.error is None and i.predicted is None for i in items),
        truncated=sum(i.finish_reason == "length" for i in items),
        accuracy=round(correct / n, 4),
        wall_s=round(wall, 2),
        gpu=gpu_snapshot(nvml=nvml),
        items=list(items),
    )
