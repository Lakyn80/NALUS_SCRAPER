from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.llm.config import effective_llm_config_from_env  # noqa: E402
from app.rag.llm.providers._base import LLMProviderError  # noqa: E402
from app.rag.llm.providers.deepseek import DeepSeekTextLLM  # noqa: E402

_ENDPOINT = "https://api.deepseek.com/chat/completions"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safe DeepSeek direct/provider smoke.")
    parser.add_argument("--mode", choices=("direct", "provider"), default="direct")
    parser.add_argument("--prompt", default="Reply with OK.")
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--json-response", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = effective_llm_config_from_env()
    print(json.dumps({"config": config.to_safe_dict()}, ensure_ascii=False))
    if not config.api_key_configured:
        print(json.dumps({"status": "blocked", "reason": "LLM_API_KEY not configured"}))
        return 2
    if args.mode == "direct":
        return _direct_smoke(args, config.deepseek_model)
    return _provider_smoke(args, config.deepseek_model, config.timeout_seconds)


def _direct_smoke(args: argparse.Namespace, model: str) -> int:
    api_key = os.getenv("LLM_API_KEY", "").strip()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
    }
    if args.json_response:
        payload["response_format"] = {"type": "json_object"}
    print(
        json.dumps(
            {
                "request": {
                    "provider": "deepseek",
                    "model": model,
                    "payload_fields": sorted(payload),
                    "message_count": len(payload["messages"]),
                    "message_lengths": [len(item["content"]) for item in payload["messages"]],
                    "max_tokens": args.max_tokens,
                    "temperature": 0.0,
                }
            },
            ensure_ascii=False,
        )
    )
    response = httpx.post(
        _ENDPOINT,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=effective_llm_config_from_env().timeout_seconds,
    )
    result = {
        "status_code": response.status_code,
        "request_id": _request_id(response.headers),
        "body": _safe_response_body(response),
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0 if 200 <= response.status_code < 300 else 1


def _provider_smoke(args: argparse.Namespace, model: str, timeout: float) -> int:
    try:
        text = DeepSeekTextLLM(
            os.getenv("LLM_API_KEY", "").strip(),
            model=model,
            timeout=timeout,
            max_tokens=args.max_tokens,
            raise_on_error=True,
            json_response=args.json_response,
        ).generate_text(args.prompt)
    except LLMProviderError as exc:
        print(json.dumps({"status": "provider_error", "error": exc.to_safe_dict()}, ensure_ascii=False))
        return 1
    print(
        json.dumps(
            {
                "status": "ok",
                "provider": "deepseek",
                "model": model,
                "output_length": len(text),
                "output_preview": text[:120],
            },
            ensure_ascii=False,
        )
    )
    return 0


def _safe_response_body(response: httpx.Response) -> dict[str, Any] | str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:600]
    if isinstance(body, dict) and "error" in body:
        error = body.get("error")
        if isinstance(error, dict):
            return {
                "error": {
                    "message": str(error.get("message") or "")[:600],
                    "type": error.get("type"),
                    "code": error.get("code"),
                    "param": error.get("param"),
                }
            }
    if isinstance(body, dict):
        choices = body.get("choices")
        content = ""
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    content = str(message.get("content") or "")
        return {
            "id": body.get("id"),
            "model": body.get("model"),
            "choice_count": len(choices) if isinstance(choices, list) else 0,
            "first_content_preview": content[:120],
            "usage": body.get("usage"),
        }
    return str(body)[:600]


def _request_id(headers: httpx.Headers) -> str | None:
    for key in ("x-request-id", "x-ds-request-id", "request-id"):
        value = headers.get(key)
        if value:
            return value[:120]
    return None


if __name__ == "__main__":
    raise SystemExit(main())
