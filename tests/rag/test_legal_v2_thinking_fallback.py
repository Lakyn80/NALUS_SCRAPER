"""Unit coverage for thinking-fallback escalation criteria."""

from app.rag.legal_v2.pipeline import _should_escalate_to_thinking_verifier
from app.rag.legal_v2.verifier import SemanticVerifierResult, VerificationDecision


def _result(**diagnostics):
    return SemanticVerifierResult(
        document_id="DOC-1",
        decision=VerificationDecision.AMBIGUOUS,
        constraint_results=[],
        raw_diagnostics=diagnostics,
    )


def test_escalate_partial_match() -> None:
    assert _should_escalate_to_thinking_verifier(
        _result(classification="partial_match", failed_closed=False)
    )


def test_escalate_insufficient_evidence() -> None:
    assert _should_escalate_to_thinking_verifier(
        _result(classification="insufficient_evidence", failed_closed=False)
    )


def test_do_not_escalate_failed_closed() -> None:
    assert not _should_escalate_to_thinking_verifier(
        _result(classification="partial_match", failed_closed=True, reason="verifier_unknown_evidence_id")
    )


def test_escalate_timeout_fail_closed() -> None:
    assert _should_escalate_to_thinking_verifier(
        _result(
            failed_closed=True,
            reason="verifier_provider_error:timeout:none:unknown",
        )
    )


def test_escalate_network_fail_closed() -> None:
    assert _should_escalate_to_thinking_verifier(
        _result(
            failed_closed=True,
            reason="verifier_provider_error:network_error:none:unknown",
        )
    )


def test_do_not_escalate_clear_strong_match() -> None:
    assert not _should_escalate_to_thinking_verifier(
        _result(classification="strong_match", failed_closed=False)
    )
