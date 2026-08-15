from pyantra_guard import PIIRedactor, redact, redact_run, redact_value

from pyantra import Run, RunEvent, RunStatus


def test_redact_default_email_and_phone() -> None:
    text = "Contact alice@example.com or 555-123-4567."
    out = redact(text)
    assert "<email>" in out
    assert "<phone>" in out
    assert "alice@example.com" not in out
    assert "555-123-4567" not in out


def test_redact_custom_patterns() -> None:
    out = redact("token abc123", patterns={"token": r"abc\d+"})
    assert out == "token <token>"


def test_redact_replaces_every_match() -> None:
    text = "a@a.com and b@b.com"
    assert redact(text) == "<email> and <email>"


def test_redact_value_nested() -> None:
    value = {
        "user": {"email": "jane@example.com", "age": 30},
        "tags": ["x@y.com", 42],
        "pair": ("p@q.com", True),
    }
    out = redact_value(value)
    assert out["user"]["email"] == "<email>"
    assert out["tags"][0] == "<email>"
    assert out["pair"][0] == "<email>"
    assert out["user"]["age"] == 30
    assert out["tags"][1] == 42
    assert out["pair"][1] is True


def test_redact_value_passthrough_scalars() -> None:
    assert redact_value(42) == 42
    assert redact_value(None) is None


def test_redact_run_events_and_interrupt() -> None:
    run = Run(
        run_id="r1",
        status=RunStatus.COMPLETED,
        state={"email": "s@example.com"},
        events=[
            RunEvent(
                run_id="r1",
                event="node",
                timestamp=1.0,
                node="n1",
                message="Reached s@example.com",
            )
        ],
        interrupt={"email": "s@example.com"},
    )
    out = redact_run(run)
    assert out.events[0].message == "Reached <email>"
    assert out.interrupt == {"email": "<email>"}
    assert out.state == {"email": "s@example.com"}
    assert run.events[0].message == "Reached s@example.com"


def test_redact_run_keeps_identity_of_original() -> None:
    run = Run(run_id="r1", status=RunStatus.COMPLETED, interrupt="a@b.com")
    out = redact_run(run)
    assert out is not run
    assert run.interrupt == "a@b.com"


def test_redactor_accepts_empty_patterns() -> None:
    redactor = PIIRedactor(patterns={})
    assert redactor.redact("a@b.com") == "a@b.com"
