"""Production Observability & Tracing via Langfuse (Graceful No-Op Fallback)."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

_CLIENT: Any = None
_INITIALIZED: bool = False


def get_langfuse_client() -> Any:
    """Return Langfuse client if configured in .env, otherwise None."""
    global _CLIENT, _INITIALIZED
    if _INITIALIZED:
        return _CLIENT

    pk = os.getenv("LANGFUSE_PUBLIC_KEY")
    sk = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if pk and sk:
        try:
            from langfuse import Langfuse
            _CLIENT = Langfuse(public_key=pk, secret_key=sk, host=host)
            logger.info("Langfuse observability initialized successfully.")
        except ImportError:
            logger.warning("Langfuse package not installed. Tracing disabled.")
            _CLIENT = None
        except Exception as e:
            logger.warning(f"Failed to initialize Langfuse: {e}")
            _CLIENT = None
    else:
        _CLIENT = None

    _INITIALIZED = True
    return _CLIENT


class RAGTracer:
    """Telemetry context for citizen RAG queries and feedback."""

    @classmethod
    def trace_query(
        cls,
        question: str,
        retrieved_chunks: list[dict],
        answer: str,
        latency_seconds: float,
        model_name: str = "gemini-3.1-pro-preview",
        user_session_id: str | None = None,
    ) -> str | None:
        client = get_langfuse_client()
        if not client:
            return None

        try:
            trace = client.trace(
                name="citizen_rag_query",
                user_id=user_session_id or "anonymous_citizen",
                input={"question": question},
                output={"answer": answer},
                metadata={
                    "latency_sec": latency_seconds,
                    "retrieved_chunk_count": len(retrieved_chunks),
                    "model": model_name,
                },
            )

            # Log retrieval span
            trace.span(
                name="retrieval_hybrid_rrf",
                input={"query": question},
                output={"chunks": [c.get("chunk_id") for c in retrieved_chunks]},
            )

            # Log generation generation
            trace.generation(
                name="rag_synthesis",
                model=model_name,
                input=question,
                output=answer,
            )

            return trace.id
        except Exception as e:
            logger.debug(f"Langfuse query trace error: {e}")
            return None

    @classmethod
    def record_feedback(
        cls,
        trace_id: str,
        score: float,  # 1.0 for thumbs up, 0.0 for thumbs down
        comment: str | None = None,
    ) -> None:
        client = get_langfuse_client()
        if not client:
            return

        try:
            client.score(
                trace_id=trace_id,
                name="user_satisfaction",
                value=score,
                comment=comment,
            )
        except Exception as e:
            logger.debug(f"Langfuse feedback score error: {e}")
