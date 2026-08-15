"""Dataset-driven batch evaluation for Pyantra runs.

A :class:`SuiteRunner` replays an app against every case in an
:class:`EvalDataset`, evaluates each :class:`~pyantra.Run` against one or more
evaluators (expectations and metrics), and aggregates the verdicts into a
:class:`SuiteReport` with pass rate, mean score, and total usage.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, TypeAlias, cast

from pyantra import Usage
from pyantra_eval.metrics import (
    EvalReport,
    EvalRun,
    Evaluator,
    _usage_to_dict,
    evaluate,
)

AppFn: TypeAlias = Callable[[Any], EvalRun | Awaitable[EvalRun]]


@dataclass(frozen=True)
class EvalCase:
    """A single input/expected pair evaluated by a suite."""

    id: str
    input: Any
    expected: Any | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "input": self.input,
            "expected": self.expected,
            "metadata": self.metadata,
        }


@dataclass
class EvalDataset:
    """A named, ordered collection of :class:`EvalCase`."""

    name: str
    cases: list[EvalCase] = field(default_factory=list)

    def add(self, case: EvalCase) -> EvalDataset:
        """Append ``case`` and return ``self`` for chaining."""
        self.cases.append(case)
        return self

    def __iter__(self) -> Iterator[EvalCase]:
        return iter(self.cases)

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> EvalCase:
        return self.cases[index]

    @classmethod
    def from_dict(
        cls, name: str, cases: Iterable[dict[str, object]]
    ) -> EvalDataset:
        """Build a dataset from a list of case dicts.

        Each dict may provide ``id`` (defaults to ``case-<index>``),
        ``input`` (required), ``expected``, and ``metadata``.
        """
        dataset = cls(name)
        for index, item in enumerate(cases):
            raw_metadata = item.get("metadata")
            metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
            dataset.add(
                EvalCase(
                    id=str(item.get("id", f"case-{index}")),
                    input=item["input"],
                    expected=item.get("expected"),
                    metadata=metadata,
                )
            )
        return dataset

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "cases": [case.to_dict() for case in self.cases]}


@dataclass(frozen=True)
class SuiteResult:
    """A single case's run together with its evaluation report."""

    case_id: str
    run: EvalRun
    report: EvalReport

    @property
    def passed(self) -> bool:
        return self.report.passed

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "run": self.run.to_dict(),
            "report": self.report.to_dict(),
        }


@dataclass(frozen=True)
class SuiteReport:
    """Aggregate verdicts across every case in a suite."""

    dataset_name: str
    results: list[SuiteResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when every case passed."""
        return all(result.passed for result in self.results)

    @property
    def failures(self) -> list[SuiteResult]:
        """The cases that did not pass."""
        return [result for result in self.results if not result.passed]

    def pass_rate(self) -> float:
        """Fraction of cases that passed; 1.0 when there are no cases."""
        if not self.results:
            return 1.0
        return sum(result.passed for result in self.results) / len(self.results)

    @property
    def avg_score(self) -> float | None:
        """Mean score across scored results, or ``None`` if there are none."""
        scores = [score for result in self.results for score in result.report.scores]
        return sum(scores) / len(scores) if scores else None

    @property
    def total_usage(self) -> Usage:
        """Aggregate LLM usage across all cases."""
        total = Usage()
        for result in self.results:
            total = total + result.report.total_usage
        return total

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset_name,
            "passed": self.passed,
            "pass_rate": self.pass_rate(),
            "avg_score": self.avg_score,
            "total_usage": _usage_to_dict(self.total_usage),
            "results": [result.to_dict() for result in self.results],
        }


class SuiteRunner:
    """Replay an app against a dataset and evaluate every run.

    ``app`` maps a case's ``input`` to a :class:`~pyantra.Run`; pass
    ``app.run`` for synchronous graphs or ``app.arun`` for async ones. A plain
    callable returning a ``Run`` (or an awaitable of one) also works. Metrics
    and expectations are evaluated per case with :func:`evaluate`.
    """

    def __init__(
        self,
        app: AppFn,
        dataset: EvalDataset,
        evaluators: Sequence[Evaluator],
        *,
        concurrency: int = 1,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self.app = app
        self.dataset = dataset
        self.evaluators = list(evaluators)
        self.concurrency = concurrency

    def run(self) -> SuiteReport:
        """Evaluate all cases synchronously, honoring ``concurrency``."""
        if self.concurrency == 1:
            results = [self._run_case(case) for case in self.dataset]
        else:
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                results = list(pool.map(self._run_case, list(self.dataset)))
        return SuiteReport(self.dataset.name, results)

    async def arun(self) -> SuiteReport:
        """Evaluate all cases asynchronously, up to ``concurrency`` at once."""
        semaphore = asyncio.Semaphore(self.concurrency)

        async def guarded(case: EvalCase) -> SuiteResult:
            async with semaphore:
                return await self._arun_case(case)

        results = await asyncio.gather(*(guarded(case) for case in self.dataset))
        return SuiteReport(self.dataset.name, list(results))

    def _run_case(self, case: EvalCase) -> SuiteResult:
        result = self.app(case.input)
        run = (
            asyncio.run(cast("Coroutine[Any, Any, EvalRun]", result))
            if inspect.isawaitable(result)
            else result
        )
        return self._assess(case.id, run)

    async def _arun_case(self, case: EvalCase) -> SuiteResult:
        result = self.app(case.input)
        run = await result if inspect.isawaitable(result) else result
        return self._assess(case.id, run)

    def _assess(self, case_id: str, run: EvalRun) -> SuiteResult:
        report = evaluate(run, *self.evaluators)
        return SuiteResult(case_id=case_id, run=run, report=report)


__all__ = [
    "AppFn",
    "EvalCase",
    "EvalDataset",
    "SuiteReport",
    "SuiteResult",
    "SuiteRunner",
]
