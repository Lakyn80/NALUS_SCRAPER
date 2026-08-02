from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class JsonExtractionDiagnostics:
    extraction_method: str
    object_start: int | None = None
    object_end: int | None = None
    direct_parse_success: bool = False
    code_fence_removed: bool = False
    prefix_length: int = 0
    suffix_length: int = 0
    ambiguity_detected: bool = False
    error: str | None = None
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JsonExtractionResult:
    payload: dict[str, Any] | None
    diagnostics: JsonExtractionDiagnostics


def extract_json_object(raw: str) -> JsonExtractionResult:
    text = raw.lstrip("\ufeff").strip()
    text, code_fence_removed = _strip_one_json_code_fence(text)

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        located = _locate_unambiguous_object(text)
        if located.error is not None:
            return JsonExtractionResult(
                payload=None,
                diagnostics=JsonExtractionDiagnostics(
                    extraction_method=located.method,
                    object_start=located.start,
                    object_end=located.end,
                    code_fence_removed=code_fence_removed,
                    prefix_length=located.prefix_length,
                    suffix_length=located.suffix_length,
                    ambiguity_detected=located.ambiguous,
                    error=located.error,
                    parse_error=_bounded_error(exc),
                ),
            )
        assert located.start is not None
        assert located.end is not None
        candidate = text[located.start : located.end]
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as nested_exc:
            return JsonExtractionResult(
                payload=None,
                diagnostics=JsonExtractionDiagnostics(
                    extraction_method=located.method,
                    object_start=located.start,
                    object_end=located.end,
                    code_fence_removed=code_fence_removed,
                    prefix_length=located.prefix_length,
                    suffix_length=located.suffix_length,
                    ambiguity_detected=located.ambiguous,
                    error="malformed_json",
                    parse_error=_bounded_error(nested_exc),
                ),
            )
        if not isinstance(payload, dict):
            return JsonExtractionResult(
                payload=None,
                diagnostics=JsonExtractionDiagnostics(
                    extraction_method=located.method,
                    object_start=located.start,
                    object_end=located.end,
                    code_fence_removed=code_fence_removed,
                    prefix_length=located.prefix_length,
                    suffix_length=located.suffix_length,
                    error="valid_json_wrong_top_level_type",
                ),
            )
        return JsonExtractionResult(
            payload=payload,
            diagnostics=JsonExtractionDiagnostics(
                extraction_method=located.method,
                object_start=located.start,
                object_end=located.end,
                code_fence_removed=code_fence_removed,
                prefix_length=located.prefix_length,
                suffix_length=located.suffix_length,
            ),
        )

    if not isinstance(payload, dict):
        return JsonExtractionResult(
            payload=None,
            diagnostics=JsonExtractionDiagnostics(
                extraction_method="direct",
                direct_parse_success=True,
                code_fence_removed=code_fence_removed,
                error="valid_json_wrong_top_level_type",
            ),
        )
    return JsonExtractionResult(
        payload=payload,
        diagnostics=JsonExtractionDiagnostics(
            extraction_method="direct",
            object_start=0,
            object_end=len(text),
            direct_parse_success=True,
            code_fence_removed=code_fence_removed,
        ),
    )


@dataclass(frozen=True)
class _LocatedObject:
    method: str
    start: int | None = None
    end: int | None = None
    prefix_length: int = 0
    suffix_length: int = 0
    ambiguous: bool = False
    error: str | None = None


def _strip_one_json_code_fence(text: str) -> tuple[str, bool]:
    lines = text.splitlines()
    if len(lines) < 2:
        return text, False
    first = lines[0].strip().casefold()
    last = lines[-1].strip()
    if first in {"```", "```json"} and last == "```":
        return "\n".join(lines[1:-1]).strip(), True
    return text, False


def _locate_unambiguous_object(text: str) -> _LocatedObject:
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            break
        end = _balanced_object_end(text, start)
        if end is None:
            return _LocatedObject("balanced_object", start=start, error="truncated_json")
        spans.append((start, end))
        index = end
    if not spans:
        return _LocatedObject("balanced_object", error="json_object_not_found")
    if len(spans) > 1:
        return _LocatedObject(
            "balanced_object",
            start=spans[0][0],
            end=spans[0][1],
            prefix_length=spans[0][0],
            suffix_length=max(0, len(text) - spans[0][1]),
            ambiguous=True,
            error="multiple_ambiguous_json_objects",
        )
    start, end = spans[0]
    prefix = text[:start].strip()
    suffix = text[end:].strip()
    if _unsafe_trailing_text(suffix):
        return _LocatedObject(
            "balanced_object",
            start=start,
            end=end,
            prefix_length=len(prefix),
            suffix_length=len(suffix),
            error="unsafe_trailing_text",
        )
    method = "prose_wrapped_json" if prefix or suffix else "balanced_object"
    return _LocatedObject(
        method,
        start=start,
        end=end,
        prefix_length=len(prefix),
        suffix_length=len(suffix),
    )


def _balanced_object_end(text: str, start: int) -> int | None:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _unsafe_trailing_text(suffix: str) -> bool:
    if not suffix:
        return False
    lowered = suffix.casefold()
    unsafe_markers = (
        "ignore previous",
        "instruction",
        "system:",
        "assistant:",
        "user:",
        "<script",
        "```",
    )
    return any(marker in lowered for marker in unsafe_markers)


def _bounded_error(exc: json.JSONDecodeError) -> str:
    return f"{exc.msg} at line {exc.lineno} column {exc.colno} char {exc.pos}"
