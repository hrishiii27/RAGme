"""Lightweight Query Router (Plain Python, No LangGraph).

Analyzes incoming user queries and routes them to optimal retrieval strategies:
- DIRECT_FACTUAL: Exact statutory sections, acts, or portal lookups -> Fast BM25 priority.
- MULTI_HOP: Comparative or multi-topic questions -> Sub-query decomposition.
- OUT_OF_DOMAIN: Greetings, chit-chat, or queries unrelated to Indian governance -> Immediate response.
- HYBRID_STANDARD: General policy and thematic questions -> Standard Hybrid RRF.
"""

from __future__ import annotations

import enum
import re
from typing import Any

from rag.llm.client import LLMClient


class RouteType(str, enum.Enum):
    DIRECT_FACTUAL = "direct_factual"
    MULTI_HOP = "multi_hop"
    OUT_OF_DOMAIN = "out_of_domain"
    HYBRID_STANDARD = "hybrid_standard"


ROUTER_PROMPT = """You are a query routing specialist for Bharat-RAG, an AI system answering questions from official Indian Government public documents.

Classify the following user query into exactly ONE of these four categories:
1. DIRECT_FACTUAL: Specific factual lookup (e.g. exact section number, portal URL, officer title, specific penalty, scheme launch date).
2. MULTI_HOP: Complex query requiring comparison between two acts/schemes, multi-part questions, or contrasting provisions across ministries.
3. OUT_OF_DOMAIN: Casual greetings, programming queries, general world facts, or questions completely unrelated to Indian government/laws/schemes.
4. HYBRID_STANDARD: General explanatory questions about Indian policies, schemes, procedures, or citizen rights.

Query: "{query}"

Output ONLY a JSON object with two fields:
{{"route": "DIRECT_FACTUAL" | "MULTI_HOP" | "OUT_OF_DOMAIN" | "HYBRID_STANDARD", "reason": "short explanation"}}"""


class QueryRouter:
    """Fast two-tier query router: regex fast-path + LLM classification."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm

        # Fast heuristic patterns
        self.greeting_pattern = re.compile(
            r"^(?:(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|namaste|vanakkam|who are you|what can you do|how are you)[\s,!\.\?]*)+$",
            re.IGNORECASE,
        )
        self.multi_hop_pattern = re.compile(
            r"\b(compare|difference between|versus|vs\.?|how does .* compare to|both .* and|contrast)\b",
            re.IGNORECASE,
        )
        self.factual_pattern = re.compile(
            r"\b(section \d+|article \d+|clause \d+|portal url|official website|helpline number|penalty for|what form|notification no)\b",
            re.IGNORECASE,
        )

    def route(self, query: str) -> tuple[RouteType, str]:
        """Routes a query to its optimal retrieval strategy."""
        clean_q = query.strip()
        if not clean_q:
            return RouteType.OUT_OF_DOMAIN, "Empty query"

        # Tier 1: Fast regex heuristics (0ms latency)
        if self.greeting_pattern.match(clean_q):
            return RouteType.OUT_OF_DOMAIN, "Heuristic match: Greeting or chit-chat"

        if self.multi_hop_pattern.search(clean_q):
            return RouteType.MULTI_HOP, "Heuristic match: Comparative / multi-entity query"

        if self.factual_pattern.search(clean_q):
            return RouteType.DIRECT_FACTUAL, "Heuristic match: Direct statutory/factual identifier"

        # Tier 2: LLM Classifier (if available for subtle distinctions)
        if self.llm is not None:
            try:
                import json
                resp = self.llm.generate(ROUTER_PROMPT.format(query=clean_q)).strip()
                # Extract JSON if enclosed in markdown
                match = re.search(r"\{.*\}", resp, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                    r_str = data.get("route", "").upper()
                    reason = data.get("reason", "LLM classified")
                    if r_str == "DIRECT_FACTUAL":
                        return RouteType.DIRECT_FACTUAL, reason
                    elif r_str == "MULTI_HOP":
                        return RouteType.MULTI_HOP, reason
                    elif r_str == "OUT_OF_DOMAIN":
                        return RouteType.OUT_OF_DOMAIN, reason
                    elif r_str == "HYBRID_STANDARD":
                        return RouteType.HYBRID_STANDARD, reason
            except Exception:
                pass

        # Default fallback
        return RouteType.HYBRID_STANDARD, "Default fallback to hybrid retrieval"
