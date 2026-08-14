"""Unit tests for adaptive token-budget packing (order restore)."""

from __future__ import annotations

from scripts.legal_v2.build_full_corpus_ab_indexes_v1 import (
    _pack_stats,
    pack_indices_by_token_budget,
)


def test_pack_indices_respects_budget_and_restores_order_membership() -> None:
    token_counts = [100, 5000, 200, 8000, 150, 300]
    packs = pack_indices_by_token_budget(
        token_counts, token_budget=10000, max_items=4
    )
    flat = [idx for pack in packs for idx in pack]
    assert sorted(flat) == list(range(len(token_counts)))
    for pack in packs:
        assert pack == sorted(pack)
        assert len(pack) <= 4
        lengths = [token_counts[i] for i in pack]
        max_seq = max(lengths)
        # Over-budget single items are allowed alone.
        if len(pack) > 1:
            assert sum(lengths) <= 10000
            assert len(pack) * max_seq <= 10000


def test_pack_indices_extreme_item_alone() -> None:
    token_counts = [50, 50, 20000, 50]
    packs = pack_indices_by_token_budget(
        token_counts, token_budget=1000, max_items=8
    )
    assert any(pack == [2] for pack in packs)
    flat = [idx for pack in packs for idx in pack]
    assert sorted(flat) == [0, 1, 2, 3]


def test_pack_does_not_mix_short_with_extreme_long() -> None:
    # 100-token shorts vs 7000-token longs live in different length buckets.
    token_counts = [100, 120, 90, 7000, 7100]
    packs = pack_indices_by_token_budget(
        token_counts, token_budget=50000, max_items=8
    )
    for pack in packs:
        lengths = [token_counts[i] for i in pack]
        assert not (min(lengths) <= 256 and max(lengths) >= 4096)


def test_pack_is_deterministic() -> None:
    token_counts = [120, 800, 450, 2100, 90, 3000, 512, 1024]
    a = pack_indices_by_token_budget(token_counts, token_budget=4096, max_items=8)
    b = pack_indices_by_token_budget(token_counts, token_budget=4096, max_items=8)
    assert a == b


def test_vector_order_restoration_after_packing() -> None:
    """Scheduler may reorder packs; restored vectors must map to original indices."""
    labels = ["A", "B", "C", "D"]
    token_counts = [100, 7000, 120, 6500]
    packs = pack_indices_by_token_budget(
        token_counts, token_budget=20000, max_items=8
    )
    # Simulate encode producing vectors labeled by original index.
    fake_vectors = {idx: f"vector({labels[idx]})" for idx in range(len(labels))}
    out: list[str | None] = [None] * len(labels)
    for pack in packs:
        for idx in pack:
            out[idx] = fake_vectors[idx]
    assert out == ["vector(A)", "vector(B)", "vector(C)", "vector(D)"]


def test_pack_stats_padded_tokens() -> None:
    token_counts = [10, 20, 30]
    stats = _pack_stats(token_counts, [0, 1, 2])
    assert stats["batch_items"] == 3
    assert stats["sum_tokens"] == 60
    assert stats["max_sequence_length"] == 30
    assert stats["padded_tokens"] == 90


def test_fixed_and_adaptive_scheduler_apis_exist() -> None:
    """Fixed batching must remain available alongside adaptive packing."""
    from scripts.legal_v2 import build_full_corpus_ab_indexes_v1 as mod

    assert hasattr(mod.GpuBgeM3Embedder, "embed_texts")
    assert hasattr(mod.GpuBgeM3Embedder, "embed_texts_adaptive")
    assert callable(mod.pack_indices_by_token_budget)
