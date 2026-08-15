"""pyantra-eval command-line interface.

``pyantra-eval run <suite.py>`` loads a suite module, runs it, prints a
per-metric summary, writes a JSON report, and exits non-zero on failure.
``pyantra-eval run <paths...>`` delegates to pytest and aggregates results via
the pyantra_eval plugin.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import importlib.util
import inspect
import json
import sys
from collections.abc import Coroutine, Sequence
from pathlib import Path
from typing import Any, cast

from pyantra_eval.judge import DiskCache, VerdictCache
from pyantra_eval.report import format_suite_report
from pyantra_eval.suite import SuiteRunner

DEFAULT_CACHE_DIR = ".pyantra-eval-cache"
DEFAULT_REPORT = "pyantra-eval-report.json"


def _load_runner(path: Path) -> SuiteRunner:
    """Import a suite module and return its configured runner."""
    spec = importlib.util.spec_from_file_location("pyantra_suite", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"error: cannot load suite module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    builder = getattr(module, "build_suite", None)
    if builder is not None:
        result: Any = builder()
        if inspect.isawaitable(result):
            result = asyncio.run(cast("Coroutine[Any, Any, Any]", result))
        runner = result
    else:
        runner = getattr(module, "suite", None)
        if runner is None:
            raise SystemExit(
                "error: suite module must define build_suite() or suite"
            )
    if not isinstance(runner, SuiteRunner):
        raise SystemExit("error: build_suite()/suite must return a SuiteRunner")
    return runner


def _rebuild_runner(
    runner: SuiteRunner,
    *,
    cache: VerdictCache | None,
    concurrency: int,
) -> SuiteRunner:
    """Rebuild ``runner`` with a cache injected into cache-aware evaluators."""
    evaluators = []
    for evaluator in runner.evaluators:
        if dataclasses.is_dataclass(evaluator) and hasattr(evaluator, "cache"):
            evaluators.append(dataclasses.replace(evaluator, cache=cache))
        else:
            evaluators.append(evaluator)
    return SuiteRunner(
        runner.app,
        runner.dataset,
        evaluators,
        concurrency=concurrency,
    )


def _run_suite(
    path: Path,
    *,
    cache: VerdictCache | None,
    concurrency: int,
    report_path: Path,
) -> int:
    runner = _rebuild_runner(_load_runner(path), cache=cache, concurrency=concurrency)
    report = runner.run()
    print(format_suite_report(report))
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8"
    )
    print(f"report written to {report_path}")
    return 0 if report.passed else 1


def _run_pytest(targets: Sequence[str], report_path: Path | None) -> int:
    import pytest

    args = list(targets)
    if report_path is not None:
        args.append(f"--pyantra-report={report_path}")
    return pytest.main(args)


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``pyantra-eval`` argument parser."""
    parser = argparse.ArgumentParser(prog="pyantra-eval")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser(
        "run", help="run a suite file (suite.py) or pytest test paths"
    )
    run_parser.add_argument(
        "target",
        nargs="+",
        help="a .py suite module, or pytest paths when not a file",
    )
    run_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="disable the on-disk judge verdict cache",
    )
    run_parser.add_argument(
        "--report",
        default=None,
        metavar="PATH",
        help=f"JSON report path (default {DEFAULT_REPORT})",
    )
    run_parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="max cases run in parallel (suite mode)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = build_parser().parse_args(argv)
    target = Path(args.target[0])

    if target.is_file() and target.suffix == ".py":
        if args.concurrency < 1:
            print("error: --concurrency must be >= 1", file=sys.stderr)
            return 2
        cache = None if args.no_cache else DiskCache(DEFAULT_CACHE_DIR)
        return _run_suite(
            target,
            cache=cache,
            concurrency=args.concurrency,
            report_path=Path(args.report or DEFAULT_REPORT),
        )

    report_path = Path(args.report) if args.report is not None else None
    return _run_pytest(args.target, report_path)


if __name__ == "__main__":
    sys.exit(main())
