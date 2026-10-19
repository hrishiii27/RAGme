"""Abstract Base Interface for RAG Vector Stores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rag.types import Chunk


class BaseVectorStore(ABC):
    """Abstract Vector Store contract supporting Local, Containerized, and Cloud Backends."""

    @abstractmethod
    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Add chunks and precomputed dense embeddings to the vector database."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search the store and return top matching chunk records."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total indexed vectors."""
        pass
