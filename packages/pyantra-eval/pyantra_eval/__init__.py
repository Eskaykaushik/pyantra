"""pyantra-eval: trajectory evaluation, LLM judges, and a pytest plugin.

* ``evaluate`` / ``expect_*`` — assert on a ``Run``'s status, error, node
  visit order, step count, or an LLM judge's score.
* ``LLMJudge`` — score output against a rubric with any ``pyantra.LLM``.
* ``pyantra_evals`` — pytest fixture that fails a test when expectations do
  not hold.
"""

from pyantra_eval.judge import DEFAULT_PROMPT, JudgeResult, LLMJudge
from pyantra_eval.metrics import (
    AbsentExpectation,
    CallableExpectation,
    ErrorExpectation,
    EvalReport,
    EvalResult,
    Evaluator,
    InterruptExpectation,
    JudgedExpectation,
    OrderExpectation,
    StatusExpectation,
    StepExpectation,
    VisitExpectation,
    evaluate,
    expect_completed,
    expect_error,
    expect_failed,
    expect_interrupt,
    expect_judged,
    expect_max_steps,
    expect_not_visited,
    expect_ordered,
    expect_status,
    expect_that,
    expect_visited,
)
from pyantra_eval.pytest_plugin import PyantraEvalCollector

__all__ = [
    "AbsentExpectation",
    "CallableExpectation",
    "DEFAULT_PROMPT",
    "ErrorExpectation",
    "EvalReport",
    "EvalResult",
    "Evaluator",
    "InterruptExpectation",
    "JudgeResult",
    "JudgedExpectation",
    "LLMJudge",
    "OrderExpectation",
    "PyantraEvalCollector",
    "StatusExpectation",
    "StepExpectation",
    "VisitExpectation",
    "evaluate",
    "expect_completed",
    "expect_error",
    "expect_failed",
    "expect_interrupt",
    "expect_judged",
    "expect_max_steps",
    "expect_not_visited",
    "expect_ordered",
    "expect_status",
    "expect_that",
    "expect_visited",
]
