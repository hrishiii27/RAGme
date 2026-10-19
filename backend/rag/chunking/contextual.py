"""Anthropic-Style Contextual Retrieval Chunker.

Solves the 'orphan chunk' problem where individual sub-clauses lose document context,
leading to failed keyword (BM25) and semantic (Dense) search matches.

Two strategies supported:
1. 'heuristic' (Contextual-Lite): Zero latency, deterministic prepend using verified document metadata:
   "[Document: {title} | Host: {host} | Topics: {topics}]\n\n{chunk_text}"
2. 'llm' (Anthropic Contextual): Generates a succinct 1-2 sentence context situating the chunk
   within the full document body via an LLM. Cached to disk to avoid duplicate API calls.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from rag.chunking.base import Chunker
from rag.chunking.strategies import RecursiveCharacterChunker
from rag.types import Chunk, Document

ANTHROPIC_CONTEXT_PROMPT = """<document>
TITLE: {title}
{document_text}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_text}
</chunk>

Please give a short, succinct context (1 to 2 sentences) to situate this chunk within the overall document for the purposes of improving search retrieval. Answer ONLY with the succinct context and nothing else."""


class ContextualChunker(Chunker):
    """Wraps any base chunker and prepends situational document context."""

    def __init__(
        self,
        base_chunker: Chunker | None = None,
        mode: str = "heuristic",  # 'heuristic' or 'llm'
        cache_dir: str | Path | None = None,
        llm: Any = None,
    ):
        self.base_chunker = base_chunker or RecursiveCharacterChunker(1000, 100)
        self.mode = mode.lower()
        if self.mode not in ("heuristic", "llm"):
            raise ValueError(f"mode must be 'heuristic' or 'llm', got '{mode}'")
        
        self.llm = llm
        if cache_dir is None:
            root = Path(__file__).resolve().parents[3]
            self.cache_dir = root / ".cache" / "contextual"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._llm_cache: dict[str, str] = self._load_cache()

    @property
    def strategy_name(self) -> str:
        base_name = getattr(self.base_chunker, "strategy_name", "base")
        return f"contextual-{self.mode}-{base_name}"

    def _load_cache(self) -> dict[str, str]:
        cache_file = self.cache_dir / "context_cache.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        cache_file = self.cache_dir / "context_cache.json"
        try:
            cache_file.write_text(json.dumps(self._llm_cache, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:
            pass

    def _build_heuristic_context(self, doc: Document) -> str:
        """Constructs a deterministic, high-signal situational context header."""
        parts = []
        if doc.title and doc.title.strip():
            clean_title = re.sub(r"\s+", " ", doc.title).strip()
            parts.append(f"Document: {clean_title}")
        
        host = doc.metadata.get("host")
        if host:
            parts.append(f"Portal: {host}")
            
        topics = doc.metadata.get("topics")
        if topics and isinstance(topics, list) and len(topics) > 0:
            topics_str = ", ".join(str(t) for t in topics[:3])
            parts.append(f"Topics: {topics_str}")

        kind = doc.metadata.get("kind")
        if kind:
            parts.append(f"Category: {kind}")

        if not parts:
            return ""
        return f"[{' | '.join(parts)}]"

    def _build_llm_context(self, doc: Document, chunk_text: str) -> str:
        """Generates Anthropic-style situational context via LLM with disk caching."""
        chunk_hash = hashlib.sha1(f"{doc.doc_id}::{chunk_text}".encode()).hexdigest()[:16]
        if chunk_hash in self._llm_cache:
            return self._llm_cache[chunk_hash]

        if self.llm is None:
            from rag.llm.client import create_llm_client
            self.llm = create_llm_client()

        # Truncate doc if excessively long to save tokens / latency
        doc_preview = doc.text[:4000] if len(doc.text) > 4000 else doc.text
        prompt = ANTHROPIC_CONTEXT_PROMPT.format(
            title=doc.title or "Official Indian Government Document",
            document_text=doc_preview,
            chunk_text=chunk_text[:1200],
        )

        try:
            context = self.llm.generate(prompt).strip()
            # Clean up unwanted prose tags
            context = re.sub(r"^Context:\s*", "", context, flags=re.IGNORECASE)
            context = re.sub(r"\n+", " ", context).strip()
            if not context:
                context = self._build_heuristic_context(doc)
        except Exception:
            context = self._build_heuristic_context(doc)

        self._llm_cache[chunk_hash] = context
        self._unsaved_count = getattr(self, "_unsaved_count", 0) + 1
        if self._unsaved_count >= 10:
            self._save_cache()
            self._unsaved_count = 0
        return context

    def flush_cache(self) -> None:
        self._save_cache()

    def chunk(self, doc: Document) -> list[Chunk]:
        raw_chunks = self.base_chunker.chunk(doc)
        if not raw_chunks:
            return []

        contextual_chunks: list[Chunk] = []

        if self.mode == "heuristic":
            doc_context = self._build_heuristic_context(doc)
            for i, c in enumerate(raw_chunks):
                if doc_context:
                    enriched_text = f"{doc_context}\n\n{c.text}"
                else:
                    enriched_text = c.text

                contextual_chunks.append(
                    Chunk(
                        doc_id=c.doc_id,
                        chunk_id=f"{c.doc_id}::{self.strategy_name}::{i}",
                        text=enriched_text,
                        start_char=c.start_char,
                        end_char=c.end_char,
                        metadata={
                            **c.metadata,
                            "raw_text": c.text,
                            "context": doc_context,
                            "context_mode": "heuristic",
                        },
                    )
                )
        else:  # 'llm' mode
            for i, c in enumerate(raw_chunks):
                chunk_context = self._build_llm_context(doc, c.text)
                if chunk_context:
                    enriched_text = f"[{chunk_context}]\n\n{c.text}"
                else:
                    enriched_text = c.text

                contextual_chunks.append(
                    Chunk(
                        doc_id=c.doc_id,
                        chunk_id=f"{c.doc_id}::{self.strategy_name}::{i}",
                        text=enriched_text,
                        start_char=c.start_char,
                        end_char=c.end_char,
                        metadata={
                            **c.metadata,
                            "raw_text": c.text,
                            "context": chunk_context,
                            "context_mode": "llm",
                        },
                    )
                )
            self._save_cache()

        return contextual_chunks
