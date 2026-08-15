# pyantra-guard

Guards for Pyantra workflows: runtime type checks, LLM budget caps, and PII
redaction. A small, focused companion to the [`pyantra`](../../packages/pyantra-core/README.md)
core.

```bash
pip install pyantra-guard
```

## Type guards

Validate values and state against annotated types at runtime, including
`Union`, `Optional`, and container generics.

```python
from pyantra_guard import assert_state, assert_type

assert_type("42", int)                 # raises TypeGuardError
assert_state(my_state)                 # checks every dataclass field
```

`typecheck(value, expected)` returns a bool for composition; `check_state(state)`
returns the list of mismatched field names.

## Budget caps

Cap tokens and cost per run. `BudgetTracker` accumulates usage and raises
`BudgetError` the moment a cap is crossed, so nodes stop as soon as the budget
is spent.

```python
from pyantra import Usage
from pyantra_guard import Budget, BudgetTracker

budget = Budget(max_total_tokens=10_000, max_cost=1.0)
tracker = BudgetTracker(budget)

def node(state):
    usage = my_llm.call(...).usage
    tracker.record(usage)   # raises BudgetError once a cap is exceeded
    ...
```

## PII redaction

Replace email, phone, SSN, credit-card, and IPv4 matches with `<label>`
placeholders, in text or across a run trace.

```python
from pyantra_guard import redact, redact_run

redact("contact alice@example.com")   # "contact <email>"
safe_run = redact_run(run)            # copy with events/interrupt redacted
```

Customize patterns via `PIIRedactor(patterns={...})` or `redact(text, patterns=...)`.
