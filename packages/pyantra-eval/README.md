# pyantra-eval

Trajectory evaluation, LLM judges, and a pytest plugin for Pyantra. Assert on
how a workflow ran — not just whether it returned.

```bash
pip install pyantra-eval
```

## Trajectory expectations

Judge a [`Run`](https://github.com/Eskaykaushik/pyantra/tree/main/packages/pyantra-core)
against expectations covering status, errors, node visits, ordering, and step
count. `evaluate` runs several at once and returns an `EvalReport`.

```python
from pyantra_eval import (
    evaluate,
    expect_completed,
    expect_ordered,
    expect_visited,
)

report = evaluate(
    run,
    expect_completed(),
    expect_visited("fetch"),
    expect_ordered("fetch", "parse", "store"),
)
assert report.passed
```

Available expectations:

- `expect_status(...)` / `expect_completed()` / `expect_failed(contains=..., pattern=...)`
- `expect_visited(node, at_least=...)` / `expect_not_visited(node)`
- `expect_ordered(*nodes)`
- `expect_max_steps(limit)`
- `expect_interrupt()`
- `expect_that(check, message=..., name=...)` — arbitrary predicates

## LLM judges

Score output against a rubric with any [`pyantra.LLM`](https://github.com/Eskaykaushik/pyantra/tree/main/packages/pyantra-core)
provider. `LLMJudge` prompts the model, parses a numeric verdict (`Score: 3`
or `Score: 7/10`), clamps it to your scale, and records the rationale and usage.

```python
from pyantra import MockLLM
from pyantra_eval import LLMJudge, expect_judged

judge = LLMJudge(MockLLM(responses=["Score: 8/10\nRationale: Accurate."]),
                 rubric="Factual accuracy", max_score=10.0)

verdict = judge.judge(answer)            # JudgeResult(score, rationale, usage)
report = evaluate(run, expect_judged(judge, threshold=6.0))
```

`expect_judged` extracts the text to score from the run (default: the final
state) and passes the verdict through the report.

## pytest plugin

Declare expectations inside a test via the `pyantra_evals` fixture; the test
fails at teardown with a summary if any expectation did not hold.

```python
def test_order_flow(pyantra_evals):
    run = app.run(state)
    pyantra_evals.expect(run, expect_completed(), expect_ordered("a", "b", "c"))
```
