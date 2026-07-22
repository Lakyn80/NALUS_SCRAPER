from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel
from starlette.datastructures import Headers

from app.core.redaction import REDACTED, fingerprint_idempotency_key, redact_sensitive


class SecretModel(BaseModel):
    api_key: str
    nested: dict[str, str]


@dataclass
class SecretDataclass:
    password: str
    public: str


def test_nested_redaction_does_not_mutate_source() -> None:
    source = {
        "user": "alice",
        "nested": {
            "client_secret": "secret-value",
            "safe": "visible",
        },
    }

    redacted = redact_sensitive(source)

    assert redacted["nested"]["client_secret"] == REDACTED
    assert redacted["nested"]["safe"] == "visible"
    assert source["nested"]["client_secret"] == "secret-value"


def test_list_and_tuple_redaction() -> None:
    payload = [
        {"Authorization": "Bearer token-value"},
        ({"refresh-token": "refresh-value"}, "safe"),
    ]

    redacted = redact_sensitive(payload)

    assert redacted[0]["Authorization"] == REDACTED
    assert redacted[1][0]["refresh-token"] == REDACTED
    assert redacted[1][1] == "safe"


def test_pydantic_model_redaction() -> None:
    model = SecretModel(
        api_key="key-value",
        nested={"database_url": "postgres://user:pass@example/db"},
    )

    redacted = redact_sensitive(model)

    assert redacted["api_key"] == REDACTED
    assert redacted["nested"]["database_url"] == REDACTED


def test_dataclass_redaction() -> None:
    payload = SecretDataclass(password="pw-value", public="ok")

    redacted = redact_sensitive(payload)

    assert redacted["password"] == REDACTED
    assert redacted["public"] == "ok"


def test_http_header_redaction() -> None:
    headers = Headers(
        {
            "authorization": "Bearer token-value",
            "cookie": "session=secret",
            "x-safe": "ok",
        }
    )

    redacted = redact_sensitive(headers)

    assert redacted["authorization"] == REDACTED
    assert redacted["cookie"] == REDACTED
    assert redacted["x-safe"] == "ok"


def test_sensitive_key_detection_is_case_and_separator_insensitive() -> None:
    payload = {
        "API-Key": "one",
        "access.token": "two",
        "Proxy Authorization": "three",
        "PRIVATE_KEY": "four",
    }

    redacted = redact_sensitive(payload)

    assert set(redacted.values()) == {REDACTED}


def test_authorization_and_cookies_are_not_unredacted_in_strings() -> None:
    payload = {
        "error": "Authorization: Bearer abc123 Cookie: session=secret",
        "password_hint": "password=abc123",
    }

    redacted = redact_sensitive(payload)
    rendered = str(redacted)

    assert "Bearer abc123" not in rendered
    assert "password=abc123" not in rendered
    assert REDACTED in rendered


def test_idempotency_key_fingerprint_is_stable_and_short() -> None:
    first = fingerprint_idempotency_key("full-idempotency-key-value")
    second = fingerprint_idempotency_key("full-idempotency-key-value")

    assert first == second
    assert len(first) == 16
    assert "full-idempotency-key-value" not in first
