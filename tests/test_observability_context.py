from __future__ import annotations

import pytest

from app.core.context import (
    bind_context,
    bound_context,
    clear_context,
    generate_correlation_id,
    generate_request_id,
    get_context,
    get_correlation_id,
    get_job_id,
    get_operation_id,
    get_request_id,
    get_task_id,
    get_workflow_id,
    is_valid_context_id,
    normalize_inbound_correlation_id,
    reset_context,
    set_job_id,
    set_operation_id,
    set_task_id,
    set_workflow_id,
)


@pytest.fixture(autouse=True)
def _clear_context() -> None:
    clear_context()
    yield
    clear_context()


def test_generated_ids_are_valid() -> None:
    assert is_valid_context_id(generate_correlation_id())
    assert is_valid_context_id(generate_request_id())


def test_inbound_correlation_id_validation() -> None:
    assert normalize_inbound_correlation_id("corr-12345678") == "corr-12345678"
    assert normalize_inbound_correlation_id("bad value") is None
    assert normalize_inbound_correlation_id("short") is None


def test_bind_context_sets_safe_identifiers() -> None:
    bind_context(
        correlation_id="corr-12345678",
        request_id="req-12345678",
        operation_id="op-12345678",
        workflow_id="wf-12345678",
        job_id="job-12345678",
        task_id="task-12345678",
    )

    assert get_correlation_id() == "corr-12345678"
    assert get_request_id() == "req-12345678"
    assert get_operation_id() == "op-12345678"
    assert get_workflow_id() == "wf-12345678"
    assert get_job_id() == "job-12345678"
    assert get_task_id() == "task-12345678"


def test_bind_context_rejects_invalid_identifier() -> None:
    with pytest.raises(ValueError):
        bind_context(correlation_id="invalid value")


def test_context_binding_can_be_reset_for_workers() -> None:
    binding = bind_context(correlation_id="corr-12345678")
    assert get_correlation_id() == "corr-12345678"

    reset_context(binding)

    assert get_correlation_id() is None


def test_bound_context_context_manager_resets() -> None:
    with bound_context(correlation_id="corr-12345678", request_id="req-12345678"):
        assert get_context()["correlation_id"] == "corr-12345678"

    assert all(value is None for value in get_context().values())


def test_setters_validate_optional_workflow_ids() -> None:
    set_operation_id("op-12345678")
    set_workflow_id("wf-12345678")
    set_job_id("job-12345678")
    set_task_id("task-12345678")

    assert get_operation_id() == "op-12345678"
    assert get_workflow_id() == "wf-12345678"
    assert get_job_id() == "job-12345678"
    assert get_task_id() == "task-12345678"

    with pytest.raises(ValueError):
        set_job_id("bad id")


def test_clear_context_removes_all_fields() -> None:
    bind_context(correlation_id="corr-12345678", request_id="req-12345678")

    clear_context()

    assert all(value is None for value in get_context().values())
