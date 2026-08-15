import json
import subprocess
import sys

from pyantra import MockLLM
from pyantra_eval import (
    DiskCache,
    EvalCase,
    EvalDataset,
    GEvalMetric,
    SuiteRunner,
    TaskCompletionMetric,
)
from pyantra_eval.cli import _rebuild_runner

_PASS_SUITE = '''
from pyantra import MockLLM, Run, RunStatus
from pyantra_eval import (
    EvalCase,
    EvalDataset,
    GEvalMetric,
    SuiteRunner,
    TaskCompletionMetric,
)


def build_suite():
    dataset = (
        EvalDataset("smoke")
        .add(EvalCase("ok", input={"ok": True, "seed": 1}))
        .add(EvalCase("also-ok", input={"ok": True, "seed": 2}))
    )

    def app(state):
        return Run(
            run_id="r1",
            status=RunStatus.COMPLETED if state["ok"] else RunStatus.FAILED,
            state=state,
        )

    llm = MockLLM(responses=["Score: 8/10\\nRationale: fine."])
    return SuiteRunner(
        app,
        dataset,
        [
            TaskCompletionMetric(extract=lambda run: str(run.state)),
            GEvalMetric(llm, rubric="quality", max_score=10.0),
        ],
    )
'''

_FAIL_SUITE = _PASS_SUITE.replace("also-ok", "bad", 1).replace(
    'EvalCase("bad", input={"ok": True, "seed": 2})',
    'EvalCase("bad", input={"ok": False, "seed": 2})',
    1,
)


def _run_cli(tmp_path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pyantra_eval.cli", *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_cli_suite_passes_and_writes_report(tmp_path) -> None:
    (tmp_path / "suite.py").write_text(_PASS_SUITE, encoding="utf-8")
    result = _run_cli(tmp_path, ["run", "suite.py"])
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "pyantra-eval-report.json").read_text())
    assert report["dataset"] == "smoke"
    assert report["passed"] is True
    assert "smoke" in result.stdout


def test_cli_suite_failure_exits_nonzero(tmp_path) -> None:
    (tmp_path / "suite.py").write_text(_FAIL_SUITE, encoding="utf-8")
    result = _run_cli(tmp_path, ["run", "suite.py"])
    assert result.returncode == 1
    report = json.loads((tmp_path / "pyantra-eval-report.json").read_text())
    assert report["passed"] is False
    assert report["pass_rate"] == 0.5


def test_cli_creates_cache_dir_and_respects_no_cache(tmp_path) -> None:
    (tmp_path / "suite.py").write_text(_PASS_SUITE, encoding="utf-8")
    result = _run_cli(tmp_path, ["run", "suite.py"])
    assert result.returncode == 0
    cache_files = list((tmp_path / ".pyantra-eval-cache").glob("*.json"))
    assert len(cache_files) == 2  # one judge call per case

    no_cache_dir = tmp_path / "no-cache"
    no_cache_dir.mkdir()
    (no_cache_dir / "suite.py").write_text(_PASS_SUITE, encoding="utf-8")
    result2 = _run_cli(no_cache_dir, ["run", "suite.py", "--no-cache"])
    assert result2.returncode == 0
    assert not (no_cache_dir / ".pyantra-eval-cache").exists()


def test_cli_pytest_mode_aggregates(tmp_path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_flow.py").write_text(
        """
from pyantra import Run, RunStatus
from pyantra_eval import expect_completed


def test_ok(pyantra_evals):
    run = Run(run_id="r", status=RunStatus.COMPLETED)
    pyantra_evals.expect(run, expect_completed())
""",
        encoding="utf-8",
    )
    result = _run_cli(tmp_path, ["run", "tests", "--report", "out.json"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "pyantra-eval session summary" in result.stdout
    report = json.loads((tmp_path / "out.json").read_text())
    assert report["dataset"] == "pytest"
    assert report["passed"] is True


def test_rebuild_runner_injects_cache_and_concurrency() -> None:
    llm = MockLLM(responses=["Score: 8/10\nRationale: fine."])
    dataset = EvalDataset("d").add(EvalCase("c", input="x"))

    def app(state: object) -> None:
        return None

    runner = SuiteRunner(
        app,
        dataset,
        [GEvalMetric(llm, rubric="q", max_score=10.0), TaskCompletionMetric()],
    )
    cache = DiskCache("ignored")
    rebuilt = _rebuild_runner(runner, cache=cache, concurrency=2)
    assert rebuilt.concurrency == 2
    metric = rebuilt.evaluators[0]
    assert isinstance(metric, GEvalMetric)
    assert metric.cache is cache
    assert isinstance(rebuilt.evaluators[1], TaskCompletionMetric)
