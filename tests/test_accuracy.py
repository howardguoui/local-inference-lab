import json

import pytest

from inference_lab import report
from inference_lab.accuracy import extract_answer, fingerprint, make_problems, run_accuracy
from inference_lab.backends import Backend
from inference_lab.cli import main
from tests.conftest import FakeNvml
from tests.fake_server import ServerThread, build_app

ANSWERS = {p.question: p.answer for p in make_problems()}


def test_problem_set_is_fixed_and_answers_are_whole_numbers():
    problems = make_problems()
    assert len(problems) == 50 and len({p.question for p in problems}) == 50
    assert all(isinstance(p.answer, int) and p.answer > 0 for p in problems)
    # Pinned: a change to the templates or the RNG would make new runs incomparable with saved ones.
    assert fingerprint(problems) == "2d5ba8a71fa3"
    assert make_problems(50) == problems and make_problems(50, seed=1) != problems
    first = problems[0]
    assert first.question.startswith("A shop sells notebooks for $8 each and pens for $4 each.")
    assert first.answer == 80 - (8 * 2 + 4 * 4)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2 x 24 = 48\nAnswer: 48", 48),
        ("Answer: 7,525", 7525),
        ("Answer: $480.", 480),
        ("Answer: **-5**", -5),
        ("So the total is \\boxed{495}.", 495),
        ("The answer is 30 years.", 30),
        ("Answer: 12. Checking: 3 trays...", 12),  # the explicit answer wins over later numbers
        ("First 40, then 16, so 56", 56),  # no marker: the last number
        ("I am not sure.", None),
    ],
)
def test_extract_answer(text, expected):
    assert extract_answer(text) == expected


def _reply(question: str) -> str | None:
    """A model that solves every problem, except it gets the train wrong, rambles on the tank and crashes on ages."""
    answer = ANSWERS[question]
    if "train" in question:
        return f"Answer: {answer + 1}"
    if "tank" in question:
        return " ".join(["thinking"] * 600) + f" Answer: {answer}"
    if "father" in question:
        return None
    return f"Step by step. Answer: {answer}"


@pytest.fixture(scope="module")
def solver():
    with ServerThread(build_app(reply=_reply)) as s:
        yield s


async def test_run_accuracy_counts_correct_wrong_truncated_and_errors(solver):
    backend = Backend("vllm", "vllm", f"{solver.url}/v1")
    result = await run_accuracy(backend, n=16, concurrency=4, max_tokens=512, label="vllm-fp8kv", nvml=FakeNvml())
    # 16 problems = 2 of each of the 8 templates: 2 wrong (train), 2 truncated (tank), 2 errors (ages)
    assert (result.correct, result.truncated, result.errors, result.unparsed) == (10, 2, 2, 2)
    assert result.accuracy == 0.625 and result.model == "fake/qwen-7b" and result.label == "vllm-fp8kv"
    assert result.fingerprint == fingerprint(make_problems(16))
    assert result.gpu["name"]
    train = next(i for i in result.items if i.id == 4)
    assert (train.expected, train.predicted, train.correct) == (495, 496, False)
    ages = next(i for i in result.items if i.id == 5)
    assert ages.error.startswith("HTTP 500") and not ages.correct
    tank = next(i for i in result.items if i.id == 8)
    assert tank.finish_reason == "length" and tank.predicted is None


async def test_run_accuracy_rejects_empty_runs(solver):
    with pytest.raises(ValueError):
        await run_accuracy(Backend("vllm", "vllm", f"{solver.url}/v1"), n=0)


def test_cli_saves_the_run_and_adds_it_to_latest_md(solver, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("INFERENCE_LAB_HOME", str(tmp_path))
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "backends.yaml").write_text(
        f"backends:\n  vllm:\n    kind: vllm\n    base_url: {solver.url}/v1\n    metrics_url: null\n"
    )
    main(["accuracy", "--backend", "vllm", "--label", "vllm-fp16kv", "--problems", "8"])
    out = capsys.readouterr().out
    assert "| vllm-fp16kv | fake/qwen-7b | arith-v1 (" in out and "| 5/8 | 62% | 1 | 1 | 1 |" in out
    assert "1 replies hit --max-tokens 512" in out
    saved = list((tmp_path / "results").glob("accuracy-vllm-fp16kv-*.json"))
    assert len(saved) == 1
    data = json.loads(saved[0].read_text())
    assert data["kind"] == "accuracy" and len(data["items"]) == 8
    md = (tmp_path / "results" / "latest.md").read_text()
    assert "## Answer accuracy" in md and "| 5/8 | 62% |" in md
    # an accuracy run is not a benchmark run: the throughput tables and charts ignore it
    assert report.load_all(tmp_path / "results") == []
    assert report.write_charts(tmp_path / "results") == []


def test_report_keeps_latest_accuracy_run_per_label_next_to_benchmarks(tmp_path):
    def acc(label: str, started: str, correct: int) -> dict:
        return {
            "kind": "accuracy",
            "label": label,
            "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "started_at": started,
            "problem_set": "arith-v1",
            "fingerprint": "2d5ba8a71fa3",
            "n": 50,
            "correct": correct,
            "accuracy": correct / 50,
            "truncated": 0,
            "unparsed": 0,
            "errors": 0,
        }

    report.save(acc("vllm-fp8kv", "2026-10-26 10:00 UTC", 40), tmp_path)
    report.save(acc("vllm-fp8kv", "2026-10-26 11:00 UTC", 44), tmp_path)
    report.save(acc("vllm-fp16kv", "2026-10-26 10:30 UTC", 45), tmp_path)
    runs = report.load_accuracy(tmp_path)
    assert [(r["label"], r["correct"]) for r in runs] == [("vllm-fp16kv", 45), ("vllm-fp8kv", 44)]
    md = report.markdown([], runs)
    assert "No runs yet" not in md
    assert "| vllm-fp8kv | Qwen/Qwen2.5-7B-Instruct-AWQ | arith-v1 (2d5ba8a71fa3) | 44/50 | 88% | 0 | 0 | 0 |" in md
