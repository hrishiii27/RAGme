"""Phase 4 — Maximal Marginal Relevance (MMR). 🧑‍💻 YOUR ASSIGNMENT

MMR re-ranks results to maximize relevance while penalizing redundancy (maximizing diversity).
"""

from __future__ import annotations
 
import math
 
 
def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
 
 
def mmr_rerank(
    query_embedding: list[float],
    candidate_embeddings: list[list[float]],
    candidate_indices: list[int],
    k: int = 5,
    lambda_param: float = 0.5,
) -> list[int]:
    if len(candidate_embeddings) != len(candidate_indices):
        raise ValueError("candidate_embeddings and candidate_indices must be the same length")
    if k <= 0 or not candidate_indices:
        return []
 
    # Similarities are computed once, not on every selection round.
    relevance = [_cosine(query_embedding, e) for e in candidate_embeddings]
    remaining = list(range(len(candidate_indices)))  # positions into the candidate lists
    selected: list[int] = []
    # sim_to_selected[p] = highest similarity between candidate p and anything chosen so far
    max_sim = [0.0] * len(candidate_indices)
 
    while remaining and len(selected) < k:
        def mmr_score(p: int) -> float:
            redundancy = max_sim[p] if selected else 0.0
            return lambda_param * relevance[p] - (1 - lambda_param) * redundancy
 
        # max() keeps the first of equal scores, i.e. the earlier (better-ranked) candidate.
        best = max(remaining, key=mmr_score)
        selected.append(best)
        remaining.remove(best)
        for p in remaining:
            sim = _cosine(candidate_embeddings[p], candidate_embeddings[best])
            if sim > max_sim[p]:
                max_sim[p] = sim
 
    return [candidate_indices[p] for p in selected]