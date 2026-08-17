from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.rag.retrieval.errors import RetrievalConfigurationError
from app.rag.retrieval.qdrant_quantization import (
    QdrantQuantizationSearchPolicy,
    qdrant_quantization_policy_from_env,
)
from scripts.legal_v2.benchmark_qdrant_quantization import main as benchmark_main
from scripts.legal_v2.enable_qdrant_scalar_int8 import main as enable_main


def test_policy_defaults_to_full_precision_ignore_true() -> None:
    policy = qdrant_quantization_policy_from_env({})
    assert policy.enabled is False
    assert policy.ignore is True
    params = policy.to_search_params()
    assert params.quantization is not None
    assert params.quantization.ignore is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "FALSE"])
def test_policy_off_values_send_ignore_true(raw: str) -> None:
    policy = qdrant_quantization_policy_from_env({"NALUS_QDRANT_QUANTIZATION_ENABLED": raw})
    assert policy.ignore is True
    assert policy.to_search_params().quantization.ignore is True


@pytest.mark.parametrize("raw", ["1", "true", "yes", "on"])
def test_policy_on_values_send_ignore_false(raw: str) -> None:
    policy = qdrant_quantization_policy_from_env(
        {
            "NALUS_QDRANT_QUANTIZATION_ENABLED": raw,
            "NALUS_QDRANT_QUANTIZATION_RESCORE": "1",
            "NALUS_QDRANT_QUANTIZATION_OVERSAMPLING": "2.0",
        }
    )
    assert policy.enabled is True
    assert policy.ignore is False
    params = policy.to_search_params()
    assert params.quantization.ignore is False
    assert params.quantization.rescore is True
    assert params.quantization.oversampling == pytest.approx(2.0)


def test_invalid_enabled_flag_is_rejected() -> None:
    with pytest.raises(RetrievalConfigurationError, match="NALUS_QDRANT_QUANTIZATION_ENABLED"):
        qdrant_quantization_policy_from_env({"NALUS_QDRANT_QUANTIZATION_ENABLED": "maybe"})


def test_invalid_oversampling_is_rejected() -> None:
    with pytest.raises(RetrievalConfigurationError, match="OVERSAMPLING"):
        qdrant_quantization_policy_from_env({"NALUS_QDRANT_QUANTIZATION_OVERSAMPLING": "0.5"})


def test_enable_script_dry_run_does_not_mutate() -> None:
    client = MagicMock()
    exit_code = enable_main(
        ["--collection", "nalus_legal_paragraph_chunks_v2_chunk_ab_v8_a_current_full"],
        client=client,
    )
    assert exit_code == 0
    client.update_collection.assert_not_called()
    client.get_collection.assert_not_called()


def test_enable_script_apply_requires_matching_confirm() -> None:
    client = MagicMock()
    with pytest.raises(ValueError, match="confirm-collection"):
        enable_main(
            ["--collection", "demo", "--apply", "--confirm-collection", "other"],
            client=client,
        )
    client.update_collection.assert_not_called()


def test_enable_script_apply_calls_update_collection_once() -> None:
    client = MagicMock()
    client.get_collection.return_value = SimpleNamespace(
        status="green",
        points_count=3,
        indexed_vectors_count=3,
        optimizer_status="ok",
        config=SimpleNamespace(quantization_config=None),
    )
    exit_code = enable_main(
        ["--collection", "demo", "--apply", "--confirm-collection", "demo"],
        client=client,
    )
    assert exit_code == 0
    client.update_collection.assert_called_once()
    kwargs = client.update_collection.call_args.kwargs
    assert kwargs["collection_name"] == "demo"
    scalar = kwargs["quantization_config"].scalar
    assert str(scalar.type).lower().endswith("int8")
    assert scalar.always_ram is True
    assert scalar.quantile == pytest.approx(0.99)


def test_benchmark_script_does_not_call_update_collection() -> None:
    client = MagicMock()
    client.scroll.return_value = (
        [SimpleNamespace(vector=[0.1] * 8)],
        None,
    )
    client.query_points.return_value = SimpleNamespace(points=[SimpleNamespace(id="1")])
    exit_code = benchmark_main(
        ["--collection", "demo", "--repeats", "2", "--limit", "3"],
        client=client,
    )
    assert exit_code == 0
    client.update_collection.assert_not_called()
    assert client.query_points.call_count == 4
    first_params = client.query_points.call_args_list[0].kwargs["search_params"]
    last_params = client.query_points.call_args_list[-1].kwargs["search_params"]
    assert first_params.quantization.ignore is True
    assert last_params.quantization.ignore is False
    for call in client.query_points.call_args_list:
        assert call.kwargs["with_payload"] is True


def test_enabled_policy_diagnostics() -> None:
    policy = QdrantQuantizationSearchPolicy(enabled=True, rescore=False, oversampling=1.5)
    diag = policy.diagnostics()
    assert diag["quantization_enabled"] is True
    assert diag["quantization_ignore"] is False
    assert diag["quantization_oversampling"] == pytest.approx(1.5)
