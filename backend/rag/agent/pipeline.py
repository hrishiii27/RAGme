"""Agentic RAG Pipeline (Plain Python, No LangGraph).

Dynamic query routing, sub-query decomposition, fast-path refutations,
and multi-hop reciprocal rank fusion without heavy agent frameworks.
"""

from __future__ import annotations

import time
from typing import Any, Iterator

from rag.agent.decomposer import QueryDecomposer
from rag.agent.router import QueryRouter, RouteType
from rag.generate.prompt import (
    SYSTEM_PROMPT_FOLLOWUP,
    build_rag_prompt,
    format_context,
)
from rag.llm.client import LLMClient
from rag.pipeline import RAGPipeline
from rag.types import Chunk

OUT_OF_DOMAIN_RESPONSE = (
    "I am Bharat-RAG, an AI assistant dedicated exclusively to official Indian Government "
    "public documents, central acts, gazettes, citizen services, and statutory welfare schemes.\n\n"
    "Your question appears to be outside this domain or does not reference official Indian governance records. "
    "Please ask a question regarding Indian acts (e.g., RTI Act, DPDP Act, Companies Act), "
    "ministry schemes, or official public portals."
)


class AgenticRAGPipeline:
    """Intelligent agentic pipeline managing query routing and decomposition."""

    def __init__(self, base_pipeline: RAGPipeline, llm: LLMClient | None = None):
        self.base_pipeline = base_pipeline
        self.llm = llm or base_pipeline.llm
        self.router = QueryRouter(llm=self.llm)
        self.decomposer = QueryDecomposer(llm=self.llm)

    def answer(
        self,
        question: str,
        chat_history: str = "",
        system_prompt: str = SYSTEM_PROMPT_FOLLOWUP,
    ) -> dict[str, Any]:
        """Routes, executes, and answers incoming queries with transparent execution traces."""
        t0 = time.perf_counter()

        # Step 1: Condense question if multi-turn history exists
        standalone_q = self.base_pipeline.condense_question(chat_history, question)

        # Step 2: Route the query
        route, route_reason = self.router.route(standalone_q)
        trace: dict[str, Any] = {
            "route": route.value,
            "route_reason": route_reason,
            "standalone_question": standalone_q,
            "sub_queries": [],
            "retrieval_strategy": "",
        }

        # Route A: OUT_OF_DOMAIN -> Fast-path refusal/clarification (0 vector calls)
        if route == RouteType.OUT_OF_DOMAIN:
            trace["retrieval_strategy"] = "none (fast-path rejection)"
            trace["total_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "question": question,
                "standalone_question": standalone_q,
                "answer": OUT_OF_DOMAIN_RESPONSE,
                "retrieved_chunks": [],
                "agent_trace": trace,
            }

        # Route B: MULTI_HOP -> Sub-query decomposition + Multi-target fusion
        if route == RouteType.MULTI_HOP:
            sub_qs = self.decomposer.decompose(standalone_q)
            trace["sub_queries"] = sub_qs
            trace["retrieval_strategy"] = f"multi-hop decomposition ({len(sub_qs)} sub-queries)"

            fused_chunks, multi_hop_trace = self.decomposer.execute_multi_hop_retrieval(
                sub_queries=sub_qs,
                retriever_fn=self.base_pipeline.retrieve,
                top_k=5,
            )
            trace["subquery_doc_ids"] = multi_hop_trace

        # Route C: DIRECT_FACTUAL -> Direct specific lookup
        elif route == RouteType.DIRECT_FACTUAL:
            trace["retrieval_strategy"] = "direct factual lookup (top-k=5)"
            fused_chunks = self.base_pipeline.retrieve(standalone_q, top_k=5)

        # Route D: HYBRID_STANDARD -> Standard hybrid RRF
        else:
            trace["retrieval_strategy"] = "standard hybrid RRF (top-k=5)"
            fused_chunks = self.base_pipeline.retrieve(standalone_q, top_k=5)

        # Step 3: Format context & Generate grounded answer
        context_str = format_context(fused_chunks)
        prompt = build_rag_prompt(standalone_q, context_str)
        response_text = self.llm.generate(prompt, system_prompt=system_prompt)

        trace["total_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "question": question,
            "standalone_question": standalone_q,
            "answer": response_text,
            "retrieved_chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "text": c.text,
                    "score": s,
                }
                for c, s in fused_chunks
            ],
            "agent_trace": trace,
        }

    def answer_stream(
        self,
        question: str,
        chat_history: str = "",
        system_prompt: str = SYSTEM_PROMPT_FOLLOWUP,
    ) -> Iterator[str]:
        """Streams output while respecting query routing."""
        standalone_q = self.base_pipeline.condense_question(chat_history, question)
        route, _ = self.router.route(standalone_q)

        if route == RouteType.OUT_OF_DOMAIN:
            for word in OUT_OF_DOMAIN_RESPONSE.split(" "):
                yield word + " "
            return

        if route == RouteType.MULTI_HOP:
            sub_qs = self.decomposer.decompose(standalone_q)
            fused_chunks, _ = self.decomposer.execute_multi_hop_retrieval(
                sub_queries=sub_qs,
                retriever_fn=self.base_pipeline.retrieve,
                top_k=5,
            )
        else:
            fused_chunks = self.base_pipeline.retrieve(standalone_q, top_k=5)

        context_str = format_context(fused_chunks)
        prompt = build_rag_prompt(standalone_q, context_str)
        for token in self.llm.stream(prompt, system_prompt=system_prompt):
            yield token
