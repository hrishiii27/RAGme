"""Phase 5 — Retrieval metrics. 🧑‍💻 YOUR ASSIGNMENT

Golden labels are DOC-level (see plan.md §4), so every metric takes:
  - `retrieved`: ranked list of doc_ids (best first; may contain duplicates because
    several chunks of the same doc can be retrieved — dedupe keeping first occurrence
    BEFORE scoring)
  - `relevant`: set of doc_ids that are correct for the query

Implement until `uv run pytest tests/test_metrics.py` passes.
"""

from __future__ import annotations
 
import math
 
 
def dedupe(retrieved: list[str]) -> list[str]:
    """Remove duplicate doc_ids, preserving first-occurrence order."""
    # dict keys keep insertion order, so this keeps the first occurrence of each id.
    return list(dict.fromkeys(retrieved))
 
 
def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant docs found in the top-k (after dedupe).
    If `relevant` is empty return 0.0."""
    if not relevant or k <= 0:
        return 0.0
    top_k = dedupe(retrieved)[:k]
    found = sum(1 for doc_id in top_k if doc_id in relevant)
    return found / len(relevant)
 
 
def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """1.0 if ANY relevant doc is in the top-k (after dedupe), else 0.0."""
    if k <= 0:
        return 0.0
    return 1.0 if any(doc_id in relevant for doc_id in dedupe(retrieved)[:k]) else 0.0
 
 
def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant doc (after dedupe); 0.0 if none."""
    for rank, doc_id in enumerate(dedupe(retrieved), start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0
 
 
def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG@k (after dedupe).
    DCG = sum(rel_i / log2(i + 1)) for rank i starting at 1.
    IDCG = DCG of an ideal ranking (all relevant docs first, up to k).
    Return 0.0 if there are no relevant docs."""
    if not relevant or k <= 0:
        return 0.0
    top_k = dedupe(retrieved)[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(top_k, start=1)
        if doc_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg
 
