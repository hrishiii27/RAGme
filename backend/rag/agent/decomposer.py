"""Sub-Query Decomposition Engine (Plain Python, No LangGraph).

Splits multi-hop and comparative questions into distinct, atomic sub-queries,
executes targeted retrievals, and fuses the results to prevent partial recall.
"""

from __future__ import annotations

import json
import re
from typing import Any

from rag.llm.client import LLMClient
from rag.search.hybrid import rrf_combine
from rag.types import Chunk

DECOMPOSE_PROMPT = """You are an expert query decomposition assistant for Bharat-RAG (Indian Government Public Information System).

The user has asked a complex multi-part or comparative question.
Break this question down into 2 (or at most 3) distinct, focused, atomic search queries.
Each sub-query should target a specific statute, ministry, policy, or entity mentioned in the original question.

Original Question: "{question}"

Output ONLY a JSON array of search queries, for example:
["RTI Act 2005 penalty provisions for public information officers", "DPDP Act 2023 financial penalties for data fiduciaries"]"""


class QueryDecomposer:
    """Decomposes multi-entity questions and merges retrieval results."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm

    def decompose(self, question: str) -> list[str]:
        """Splits question into 2-3 focused sub-queries."""
        if not self.llm:
            # Fallback heuristic: split on comparative keywords
            parts = re.split(r"\b(?:and|versus|vs\.?|compared to|contrast with)\b", question, flags=re.IGNORECASE)
            sub_qs = [p.strip() for p in parts if len(p.strip()) > 5]
            return sub_qs[:3] if len(sub_qs) >= 2 else [question]

        try:
            resp = self.llm.generate(DECOMPOSE_PROMPT.format(question=question)).strip()
            match = re.search(r"\[.*\]", resp, re.DOTALL)
            if match:
                queries = json.loads(match.group(0))
                if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                    return [q.strip() for q in queries if q.strip()][:3]
        except Exception:
            pass

        return [question]

    def execute_multi_hop_retrieval(
        self,
        sub_queries: list[str],
        retriever_fn: Any,
        top_k: int = 5,
    ) -> tuple[list[tuple[Chunk, float]], dict[str, list[str]]]:
        """Executes retrieval for each sub-query and fuses results using RRF."""
        subquery_hits: dict[str, list[tuple[Chunk, float]]] = {}
        trace_info: dict[str, list[str]] = {}

        for sq in sub_queries:
            hits = retriever_fn(sq, top_k=top_k * 2)
            subquery_hits[sq] = hits
            trace_info[sq] = [c.doc_id for c, _ in hits[:3]]

        # Interleave hits round-robin across all sub-queries to guarantee balanced entity coverage
        all_hit_lists = list(subquery_hits.values())
        if not all_hit_lists or not any(all_hit_lists):
            return [], trace_info

        fused: list[tuple[Chunk, float]] = []
        seen_ids: set[str] = set()
        max_depth = max(len(h) for h in all_hit_lists)

        for depth in range(max_depth):
            for hits in all_hit_lists:
                if depth < len(hits):
                    chunk, score = hits[depth]
                    if chunk.chunk_id not in seen_ids:
                        seen_ids.add(chunk.chunk_id)
                        fused.append((chunk, score))
                        if len(fused) >= top_k:
                            return fused, trace_info

        return fused, trace_info
