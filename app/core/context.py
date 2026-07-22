"""Request and workflow correlation context.

The identifiers in this module are observability-only values. They must never
be used for authentication, authorization, ownership, or tenant decisions.
"""

from __future__ import annotations

import re
import secrets
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator
from contextvars import ContextVar, Token

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
_GENERATED_BYTES = 24

CORRELATION_ID_HEADER = "X-Correlation-ID"
REQUEST_ID_HEADER = "X-Request-ID"

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_operation_id: ContextVar[str | None] = ContextVar("operation_id", default=None)
_workflow_id: ContextVar[str | None] = ContextVar("workflow_id", default=None)
_job_id: ContextVar[str | None] = ContextVar("job_id", default=None)
_task_id: ContextVar[str | None] = ContextVar("task_id", default=None)

_CONTEXT_VARS: dict[str, ContextVar[str | None]] = {
    "correlation_id": _correlation_id,
    "request_id": _request_id,
    "operation_id": _operation_id,
    "workflow_id": _workflow_id,
    "job_id": _job_id,
    "task_id": _task_id,
}


@dataclass(frozen=True)
class ContextBinding:
    tokens: dict[str, Token[str | None]]


def generate_correlation_id() -> str:
    return f"c-{secrets.token_urlsafe(_GENERATED_BYTES)}"


def generate_request_id() -> str:
    return f"r-{secrets.token_urlsafe(_GENERATED_BYTES)}"


def is_valid_context_id(value: str | None) -> bool:
    if value is None:
        return False
    candidate = value.strip()
    return candidate == value and bool(_ID_RE.fullmatch(candidate))


def normalize_inbound_correlation_id(value: str | None) -> str | None:
    if value is None:
        return None
    candidate = value.strip()
    return candidate if is_valid_context_id(candidate) else None


def get_correlation_id() -> str | None:
    return _correlation_id.get()


def get_request_id() -> str | None:
    return _request_id.get()


def get_operation_id() -> str | None:
    return _operation_id.get()


def get_workflow_id() -> str | None:
    return _workflow_id.get()


def get_job_id() -> str | None:
    return _job_id.get()


def get_task_id() -> str | None:
    return _task_id.get()


def get_context() -> dict[str, str | None]:
    return {name: var.get() for name, var in _CONTEXT_VARS.items()}


def bind_context(
    *,
    correlation_id: str | None = None,
    request_id: str | None = None,
    operation_id: str | None = None,
    workflow_id: str | None = None,
    job_id: str | None = None,
    task_id: str | None = None,
) -> ContextBinding:
    values = {
        "correlation_id": correlation_id,
        "request_id": request_id,
        "operation_id": operation_id,
        "workflow_id": workflow_id,
        "job_id": job_id,
        "task_id": task_id,
    }
    tokens: dict[str, Token[str | None]] = {}
    for name, value in values.items():
        if value is None:
            continue
        if not is_valid_context_id(value):
            raise ValueError(f"{name} has an invalid observability identifier format.")
        tokens[name] = _CONTEXT_VARS[name].set(value)
    return ContextBinding(tokens=tokens)


def reset_context(binding: ContextBinding) -> None:
    for name, token in reversed(list(binding.tokens.items())):
        _CONTEXT_VARS[name].reset(token)


@contextmanager
def bound_context(**values: str | None) -> Iterator[None]:
    binding = bind_context(**values)
    try:
        yield
    finally:
        reset_context(binding)


def clear_context() -> None:
    for var in _CONTEXT_VARS.values():
        var.set(None)


def set_operation_id(value: str | None) -> None:
    _set_optional_context_id(_operation_id, "operation_id", value)


def set_workflow_id(value: str | None) -> None:
    _set_optional_context_id(_workflow_id, "workflow_id", value)


def set_job_id(value: str | None) -> None:
    _set_optional_context_id(_job_id, "job_id", value)


def set_task_id(value: str | None) -> None:
    _set_optional_context_id(_task_id, "task_id", value)


def _set_optional_context_id(
    var: ContextVar[str | None],
    name: str,
    value: str | None,
) -> None:
    if value is not None and not is_valid_context_id(value):
        raise ValueError(f"{name} has an invalid observability identifier format.")
    var.set(value)
