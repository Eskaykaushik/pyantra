"""PII redaction for strings, nested values, and run traces.

Matches are replaced with a ``<label>`` placeholder (e.g. ``<email>``), so
redacted traces stay readable while the raw value never leaves the process
in logged output.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any

from pyantra import Run

DEFAULT_PATTERNS: dict[str, str] = {
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "phone": r"(?:\+?\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}",
    "ssn": r"\d{3}-\d{2}-\d{4}",
    "credit_card": r"(?:\d{4}[ -]?){3}\d{4}",
    "ipv4": r"\d{1,3}(?:\.\d{1,3}){3}",
}


class PIIRedactor:
    """Redacts known PII shapes from text.

    ``patterns`` maps a label to a regex; every match is replaced with
    ``<label>``. Patterns are applied in order.
    """

    def __init__(self, patterns: dict[str, str] | None = None) -> None:
        self.patterns: dict[str, str] = (
            dict(patterns) if patterns is not None else dict(DEFAULT_PATTERNS)
        )

    def redact(self, text: str) -> str:
        """Replace all PII matches in ``text`` with ``<label>`` placeholders."""
        for label, pattern in self.patterns.items():
            text = re.sub(pattern, f"<{label}>", text)
        return text

    def redact_value(self, value: Any) -> Any:
        """Redact strings inside nested ``dict``/``list``/``tuple`` values.

        Non-string scalars pass through unchanged; nested containers are
        copied with their strings redacted.
        """
        if isinstance(value, str):
            return self.redact(value)
        if isinstance(value, dict):
            return {key: self.redact_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.redact_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.redact_value(item) for item in value)
        return value

    def redact_run(self, run: Run[Any]) -> Run[Any]:
        """Return a copy of ``run`` with event messages and interrupt payloads redacted.

        State and other fields are left untouched so the run stays usable.
        """
        events = [
            dataclasses.replace(
                event,
                message=self.redact(event.message) if event.message else None,
            )
            for event in run.events
        ]
        return dataclasses.replace(
            run,
            events=events,
            interrupt=self.redact_value(run.interrupt),
        )


def redact(text: str, *, patterns: dict[str, str] | None = None) -> str:
    """Redact PII from ``text`` using the default patterns (or custom ones)."""
    return PIIRedactor(patterns).redact(text)


def redact_value(value: Any, *, patterns: dict[str, str] | None = None) -> Any:
    """Redact PII from strings inside nested containers."""
    return PIIRedactor(patterns).redact_value(value)


def redact_run(run: Run[Any], *, patterns: dict[str, str] | None = None) -> Run[Any]:
    """Return a copy of ``run`` with event messages and interrupt payloads redacted."""
    return PIIRedactor(patterns).redact_run(run)


__all__ = [
    "DEFAULT_PATTERNS",
    "PIIRedactor",
    "redact",
    "redact_run",
    "redact_value",
]
