"""End-to-End RAG Pipeline Orchestrator."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

from rag.chunking.strategies import RecursiveCharacterChunker
from rag.embed.base import CachedEmbedder
from rag.embed.local import LocalEmbedder
from rag.generate.prompt import (
    CONDENSE_PROMPT,
    SYSTEM_PROMPT_FOLLOWUP,
    build_rag_prompt,
    format_context,
)
from rag.llm.client import LLMClient, create_llm_client
from rag.search.hybrid import rrf_combine
from rag.search.lexical import BM25Retriever
from rag.store.base import BaseVectorStore
from rag.store.factory import create_vector_store
from rag.types import Chunk, Document


def find_project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "data").exists() and (parent / "reports").exists():
            return parent
    return p.parents[3]


class RAGPipeline:
    def __init__(
        self,
        docs: list[Document],
        collection_name: str = "A6000-recursive-1000-100",
        llm: LLMClient | None = None,
        store_type: str | None = None,
    ):
        self.root = find_project_root()
        self.docs = docs
        self.chunker = RecursiveCharacterChunker(1000, 100)
        self.embedder = CachedEmbedder(
            LocalEmbedder("all-MiniLM-L6-v2"),
            cache_dir=str(self.root / ".cache" / "embeddings"),
        )
        self.store = create_vector_store(
            collection_name=collection_name,
            persist_dir=str(self.root / ".chroma"),
            store_type=store_type,
        )
        
        # Prepare all chunks and BM25 index
        print("Preparing chunks for pipeline...")
        all_chunks: list[Chunk] = []
        for d in docs:
            all_chunks.extend(self.chunker.chunk(d))
        self.all_chunks = all_chunks
        self.chunk_by_id = {c.chunk_id: c for c in all_chunks}
        
        print(f"Building BM25 index over {len(all_chunks)} chunks...")
        self.bm25 = BM25Retriever(all_chunks)
        self.llm = llm or create_llm_client()

    def condense_question(self, chat_history: str, question: str) -> str:
        """Condense conversation history + question into a standalone query."""
        if not chat_history.strip():
            return question
        prompt = CONDENSE_PROMPT.format(chat_history=chat_history, question=question)
        condensed = self.llm.generate(prompt)
        return condensed.strip() or question

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        # Dense search
        q_vec = self.embedder.embed_query(query)
        d_results = self.store.search(q_vec, n_results=top_k * 4)
        dense_hits = []
        for r in d_results:
            c = self.chunk_by_id.get(r["id"]) if self.chunk_by_id else None
            if not c and r.get("document"):
                meta = r.get("metadata", {})
                c = Chunk(
                    chunk_id=r["id"],
                    doc_id=meta.get("doc_id") or "",
                    text=r["document"],
                    start_char=int(meta.get("start_char") or 0),
                    end_char=int(meta.get("end_char") or len(r["document"])),
                    metadata={
                        k: v
                        for k, v in meta.items()
                        if k not in ("text", "doc_id", "start_char", "end_char")
                    },
                )
            if c:
                dense_hits.append((c, 1.0 - (r.get("distance") or 0.0)))

        # BM25 search
        bm25_hits = self.bm25.search(query, k=top_k * 4) if self.bm25 else []

        # RRF combine
        if bm25_hits:
            fused = rrf_combine(bm25_hits, dense_hits, k=top_k, rank_constant=60)
        else:
            fused = dense_hits[:top_k]
        return fused

    def answer(
        self,
        question: str,
        chat_history: str = "",
        system_prompt: str = SYSTEM_PROMPT_FOLLOWUP,
    ) -> dict:
        """Single-shot or multi-turn RAG answer with follow-up questions."""
        standalone_q = self.condense_question(chat_history, question)
        fused_chunks = self.retrieve(standalone_q, top_k=5)
        context_str = format_context(fused_chunks)
        prompt = build_rag_prompt(standalone_q, context_str)
        response_text = self.llm.generate(prompt, system_prompt=system_prompt)

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
        }

    def answer_stream(
        self,
        question: str,
        chat_history: str = "",
        system_prompt: str = SYSTEM_PROMPT_FOLLOWUP,
    ) -> Iterator[str]:
        """Streaming version of RAG answer generation."""
        standalone_q = self.condense_question(chat_history, question)
        fused_chunks = self.retrieve(standalone_q, top_k=5)
        context_str = format_context(fused_chunks)
        prompt = build_rag_prompt(standalone_q, context_str)
        return self.llm.stream(prompt, system_prompt=system_prompt)
