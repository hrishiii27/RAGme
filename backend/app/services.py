"""Pipeline service manager and lifecycle coordination."""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import AsyncIterator, Iterator

import pyarrow.parquet as pq

from rag.pipeline import RAGPipeline, find_project_root
from rag.types import Document
from app.schemas import SourceChunk

_PIPELINE: RAGPipeline | None = None
_DOC_COUNT: int = 0
_CHUNK_COUNT: int = 0


def get_pipeline() -> RAGPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = init_rag_system()
    return _PIPELINE


def get_stats() -> dict:
    global _DOC_COUNT, _CHUNK_COUNT
    import os
    pipeline = get_pipeline()
    store_type = os.getenv("VECTOR_STORE_TYPE", "pinecone" if os.getenv("PINECONE_API_KEY") else "chroma").lower()
    if store_type == "pinecone":
        store_desc = f"Pinecone Cloud ({os.getenv('PINECONE_INDEX_NAME', 'bharat-rag')})"
        try:
            live_count = pipeline.store.count()
            if live_count > 0:
                _CHUNK_COUNT = live_count
                _DOC_COUNT = int(live_count / 14.59)
        except Exception:
            pass
    elif store_type == "chroma":
        store_desc = (
            f"Chroma HTTP ({os.getenv('CHROMA_HOST')})" if os.getenv("CHROMA_HOST")
            else "Chroma Local (Embedded SQLite)"
        )
    else:
        store_desc = f"{store_type.capitalize()} Cloud"
    return {
        "status": "online",
        "total_documents": _DOC_COUNT,
        "total_chunks": _CHUNK_COUNT,
        "chroma_collection": "A15000-recursive-1000-100",
        "vector_store": store_desc,
        "dense_embedder": "all-MiniLM-L6-v2 (Cached 384-d)",
        "retrieval_strategy": "Hybrid RRF (BM25 + Dense, k=60)",
        "llm_model": f"{pipeline.llm.model_name} (with fallback)",
        "prompt_winner": "Few-Shot Learned (Seed Aligned, Hybrid Follow-ups)",
        "device": "Mac M2 / Container",
    }


def init_rag_system() -> RAGPipeline:
    global _PIPELINE, _DOC_COUNT, _CHUNK_COUNT
    if _PIPELINE is not None:
        return _PIPELINE

    root = find_project_root()
    parquet_path = root / "data" / "subset" / "english.parquet"

    print(f"Loading corpus from {parquet_path}...")
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    _DOC_COUNT = len(rows)

    import os
    store_type = os.getenv("VECTOR_STORE_TYPE", "pinecone" if os.getenv("PINECONE_API_KEY") else "chroma").lower()

    if store_type == "pinecone":
        # Cloud-first mode: Pinecone Cloud holds all 248,041 vectors and metadata.
        # We load a lean sample in RAM for BM25 fallback, reducing RAM by >90% and keeping M2 cool.
        _DOC_COUNT = 17000
        _CHUNK_COUNT = 248041
        lean_rows = rows[:600]
        docs = [
            Document(
                doc_id=r["doc_id"],
                title=r.get("title") or "Bharat Government Document",
                text=r.get("text") or "",
                url=r.get("url") or "",
                metadata={
                    "kind": r.get("kind", ""),
                    "topics": r.get("topics", []),
                    "trust": r.get("trust", ""),
                },
            )
            for r in lean_rows
        ]
        print("Initializing RAGPipeline in Cloud-First Mode (Pinecone Serverless)...")
        pipeline = RAGPipeline(docs=docs, collection_name="A15000-recursive-1000-100", store_type="pinecone")
    else:
        docs = [
            Document(
                doc_id=r["doc_id"],
                title=r.get("title") or "Bharat Government Document",
                text=r.get("text") or "",
                url=r.get("url") or "",
                metadata={
                    "kind": r.get("kind", ""),
                    "topics": r.get("topics", []),
                    "trust": r.get("trust", ""),
                },
            )
            for r in rows
        ]
        print("Initializing RAGPipeline (Embedder, Vector Store, BM25 Index)...")
        pipeline = RAGPipeline(docs=docs, collection_name="A15000-recursive-1000-100")
        _CHUNK_COUNT = len(pipeline.all_chunks)

    _PIPELINE = pipeline
    print(f"RAGPipeline initialized successfully (Vectors served from {store_type.upper()}).")
    return _PIPELINE


def extract_follow_ups(text: str) -> tuple[str, list[str]]:
    """Separates citizen answer and suggested follow-up questions."""
    # First strip any hidden reasoning blocks if present
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL).strip()

    marker = "### Suggested Follow-up Questions"
    if marker in text:
        parts = text.split(marker, 1)
        main_answer = parts[0].strip()
        follow_up_block = parts[1].strip()
        questions = []
        for line in follow_up_block.splitlines():
            cleaned = line.strip().lstrip("-*•123456789. ")
            # Remove any leading [Factual]: or [Actionable]: prefixes for clean chip display
            cleaned = re.sub(r"^\[.*?\]:?\s*", "", cleaned).strip()
            if cleaned and len(cleaned) > 10:
                questions.append(cleaned)
        return main_answer, questions[:2]
    return text.strip(), []


def save_feedback(data: dict) -> str:
    root = find_project_root()
    feedback_file = root / "data" / "feedback.jsonl"
    feedback_id = f"fb-{uuid.uuid4().hex[:8]}"
    record = {
        "feedback_id": feedback_id,
        "timestamp": time.time(),
        **data,
    }
    with feedback_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    # If trace_id is present, forward feedback to Langfuse
    trace_id = data.get("trace_id")
    if trace_id:
        try:
            from rag.eval.tracer import RAGTracer
            score = 1.0 if data.get("rating") == "up" else 0.0
            RAGTracer.record_feedback(
                trace_id=trace_id,
                score=score,
                comment=data.get("comment"),
            )
        except Exception:
            pass

    return feedback_id
