"""Phase 4 — Hybrid Search (RRF). 🧑‍💻 YOUR ASSIGNMENT

Reciprocal Rank Fusion (RRF) combines rankings from multiple search strategies 
(e.g., BM25 and Semantic Search) without requiring calibrated scores.
"""

from __future__ import annotations
 
from rag.types import Chunk
 
 
def rrf_combine(
    lexical_results: list[tuple[Chunk, float]],
    semantic_results: list[tuple[Chunk, float]],
    k: int = 5,
    rank_constant: int = 60,
) -> list[tuple[Chunk, float]]:
    if k <= 0:
        return []
    fused: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    first_seen: dict[str, int] = {}
 
    for results in (lexical_results, semantic_results):
        for rank, (chunk, _score) in enumerate(results, start=1):  # 1-indexed
            cid = chunk.chunk_id
            by_id.setdefault(cid, chunk)
            first_seen.setdefault(cid, len(first_seen))
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (rank_constant + rank)
 
    ranked = sorted(fused, key=lambda cid: (-fused[cid], first_seen[cid]))[:k]
    return [(by_id[cid], fused[cid]) for cid in ranked]
 
