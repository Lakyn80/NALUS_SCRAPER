"""Local sentence-transformers CrossEncoder provider for Legal v2."""

from __future__ import annotations

import math
from threading import Lock
from typing import Any, Sequence

from app.rag.legal_v2.rerank.config import CrossEncoderConfig, DEFAULT_CE_MODEL
from app.rag.legal_v2.rerank.errors import (
    RerankerInferenceError,
    RerankerModelLoadError,
)
from app.rag.legal_v2.rerank.models import RerankPassage, RerankScore


def resolve_device(preferred: str) -> str:
    pref = (preferred or "auto").strip().lower()
    if pref not in {"auto", "cpu", "cuda"}:
        pref = "auto"
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"
    # auto
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


class SentenceTransformersCrossEncoderProvider:
    """Lazy-loaded CrossEncoder. Injectable model for unit tests."""

    def __init__(
        self,
        config: CrossEncoderConfig | None = None,
        *,
        model: Any | None = None,
    ) -> None:
        self._config = config or CrossEncoderConfig()
        self._model = model
        self._loaded = model is not None
        self._device = "injected" if model is not None else "unloaded"
        self._dtype = "unknown"
        self._revision: str | None = None
        self._lock = Lock()

    @property
    def model_id(self) -> str:
        return self._config.model_id or DEFAULT_CE_MODEL

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_revision(self) -> str | None:
        return self._revision

    @property
    def dtype(self) -> str:
        return self._dtype

    def load(self) -> None:
        if self._loaded and self._model is not None:
            return
        with self._lock:
            if self._loaded and self._model is not None:
                return
            device = resolve_device(self._config.device)
            local_files_only = bool(self._config.local_files_only) and not bool(
                self._config.allow_download
            )
            try:
                from sentence_transformers import CrossEncoder  # type: ignore[import]
            except Exception as exc:  # noqa: BLE001
                raise RerankerModelLoadError(
                    "sentence-transformers CrossEncoder is unavailable"
                ) from exc
            try:
                # sentence-transformers>=5 exposes local_files_only as a top-level
                # CrossEncoder kwarg; do not also pass it via model_kwargs.
                try:
                    self._model = CrossEncoder(
                        self.model_id,
                        device=device,
                        max_length=int(self._config.max_length),
                        trust_remote_code=False,
                        local_files_only=local_files_only,
                    )
                except TypeError:
                    # Older ST signatures may omit local_files_only.
                    self._model = CrossEncoder(
                        self.model_id,
                        device=device,
                        max_length=int(self._config.max_length),
                        trust_remote_code=False,
                    )
            except Exception as exc:  # noqa: BLE001
                raise RerankerModelLoadError(
                    f"failed to load cross-encoder model {self.model_id!r}"
                ) from exc
            self._device = device
            self._dtype = "float32"
            self._revision = _try_model_revision(self._model)
            self._loaded = True

    def score(
        self,
        query: str,
        passages: Sequence[RerankPassage],
    ) -> Sequence[RerankScore]:
        if not passages:
            return ()
        cleaned_query = " ".join(str(query or "").split()).strip()
        if not cleaned_query:
            raise RerankerInferenceError("query must not be blank for CE scoring")
        self.load()
        assert self._model is not None

        batch_size = max(1, int(self._config.batch_size))
        max_len = max(32, int(self._config.max_length))
        # Rough char budget (~4 chars/token) for truncation accounting only.
        char_budget = max_len * 4
        pairs: list[list[str]] = []
        truncated_flags: list[bool] = []
        for passage in passages:
            q = cleaned_query
            p = passage.text
            truncated = False
            if len(q) > char_budget // 3:
                q = q[: char_budget // 3]
                truncated = True
            if len(p) > char_budget:
                p = p[:char_budget]
                truncated = True
            pairs.append([q, p])
            truncated_flags.append(truncated)

        raw_scores: list[float] = []
        try:
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start : start + batch_size]
                pred = self._model.predict(batch, batch_size=len(batch), show_progress_bar=False)
                if hasattr(pred, "tolist"):
                    values = pred.tolist()
                else:
                    values = list(pred)
                for value in values:
                    score = float(value)
                    if not math.isfinite(score):
                        score = 0.0
                    raw_scores.append(score)
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" in message or "cuda" in message:
                raise RerankerInferenceError("CUDA/CPU OOM during cross-encoder inference") from exc
            raise RerankerInferenceError("cross-encoder inference failed") from exc
        except Exception as exc:  # noqa: BLE001
            raise RerankerInferenceError("cross-encoder inference failed") from exc

        if len(raw_scores) != len(passages):
            raise RerankerInferenceError(
                f"score count mismatch: expected {len(passages)}, got {len(raw_scores)}"
            )

        return tuple(
            RerankScore(
                ecli=passage.ecli,
                chunk_id=passage.chunk_id,
                score=score,
                passage_index=passage.passage_index,
                truncated=flag,
            )
            for passage, score, flag in zip(passages, raw_scores, truncated_flags)
        )


def _try_model_revision(model: Any) -> str | None:
    for attr in ("model", "model_card_data", "config"):
        obj = getattr(model, attr, None)
        if obj is None:
            continue
        for key in ("_name_or_path", "name_or_path"):
            value = getattr(obj, key, None)
            if isinstance(value, str) and value.strip():
                return value.strip()[:200]
        cfg = getattr(obj, "config", None)
        if cfg is not None:
            value = getattr(cfg, "_name_or_path", None)
            if isinstance(value, str) and value.strip():
                return value.strip()[:200]
    return None
