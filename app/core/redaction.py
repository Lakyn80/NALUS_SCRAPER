"""Central sensitive-data redaction helpers."""

from __future__ import annotations

import dataclasses
import hashlib
import re
from collections.abc import Mapping
from typing import Any

REDACTED = "[REDACTED]"

_SENSITIVE_KEYS = {
    "authorization",
    "proxyauthorization",
    "cookie",
    "setcookie",
    "password",
    "passwd",
    "secret",
    "apikey",
    "accesstoken",
    "refreshtoken",
    "privatekey",
    "clientsecret",
    "databaseurl",
    "dsn",
    "idempotencykey",
}

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/\-=]+"),
    re.compile(r"(?i)(api[_\-. ]?key\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"(?i)(access[_\-. ]?token\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"(?i)(refresh[_\-. ]?token\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"(?i)(password\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"(?i)(client[_\-. ]?secret\s*[:=]\s*)[^\s,;]+"),
)


def fingerprint_idempotency_key(value: str, *, length: int = 16) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return digest[:length]


def is_sensitive_key(key: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
    if normalized in _SENSITIVE_KEYS:
        return True
    return "secret" in normalized or normalized.endswith("password")


def redact_sensitive(value: Any) -> Any:
    """Return a redacted deep copy without mutating caller-owned structures."""

    return _redact(value, seen=set())


def _redact(value: Any, *, seen: set[int]) -> Any:
    if _is_scalar(value):
        return _redact_scalar(value)

    object_id = id(value)
    if object_id in seen:
        return "[CIRCULAR]"

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        seen.add(object_id)
        return _redact(dataclasses.asdict(value), seen=seen)

    if hasattr(value, "model_dump") and callable(value.model_dump):
        seen.add(object_id)
        return _redact(value.model_dump(), seen=seen)

    if hasattr(value, "dict") and callable(value.dict):
        seen.add(object_id)
        return _redact(value.dict(), seen=seen)

    if isinstance(value, Mapping):
        seen.add(object_id)
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            redacted[key] = REDACTED if is_sensitive_key(key) else _redact(item, seen=seen)
        return redacted

    if isinstance(value, list):
        seen.add(object_id)
        return [_redact(item, seen=seen) for item in value]

    if isinstance(value, tuple):
        seen.add(object_id)
        return tuple(_redact(item, seen=seen) for item in value)

    if isinstance(value, BaseException):
        return {
            "type": type(value).__name__,
            "message": _redact_string(str(value)),
        }

    return _redact_string(str(value))


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _redact_scalar(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_string(value)
    return value


def _redact_string(value: str) -> str:
    redacted = value
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(_pattern_replacement, redacted)
    return redacted


def _pattern_replacement(match: re.Match[str]) -> str:
    if match.lastindex:
        return f"{match.group(1)}{REDACTED}"
    return REDACTED
