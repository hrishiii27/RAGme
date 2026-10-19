import math

import pytest

from rag.eval.metrics import dedupe, hit_at_k, mrr, ndcg_at_k, recall_at_k


def test_dedupe_keeps_first_occurrence():
    assert dedupe(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_recall():
    assert recall_at_k(["a", "x", "b"], {"a", "b"}, 2) == 0.5
    assert recall_at_k(["a", "x", "b"], {"a", "b"}, 3) == 1.0
    assert recall_at_k(["a"], set(), 3) == 0.0


def test_recall_dedupes_before_cutoff():
    # "a","a","a" collapses to just "a", so "b" lands at rank 2
    assert recall_at_k(["a", "a", "a", "b"], {"a", "b"}, 2) == 1.0


def test_hit():
    assert hit_at_k(["x", "y", "a"], {"a"}, 2) == 0.0
    assert hit_at_k(["x", "y", "a"], {"a"}, 3) == 1.0


def test_mrr():
    assert mrr(["x", "a"], {"a"}) == 0.5
    assert mrr(["x", "y"], {"a"}) == 0.0
    assert mrr(["x", "x", "a"], {"a"}) == 0.5  # dedupe first


def test_ndcg_perfect_and_zero():
    assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, 3) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], {"a"}, 2) == 0.0
    assert ndcg_at_k(["a"], set(), 2) == 0.0


def test_ndcg_rank_matters():
    got = ndcg_at_k(["x", "a"], {"a"}, 2)
    assert got == pytest.approx(1 / math.log2(3))
