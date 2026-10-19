"""Phase 2 — Chunking Base Interface."""

from __future__ import annotations

from typing import Protocol

from rag.types import Chunk, Document


class Chunker(Protocol):
    """Protocol for all chunking strategies."""
    
    @property
    def strategy_name(self) -> str:
        """A short identifier, e.g. 'fixed-512' or 'recursive'."""
        ...

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split a document into chunks."""
        ...
