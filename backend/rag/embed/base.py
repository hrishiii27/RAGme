"""Phase 3 — Embeddings Base."""

from __future__ import annotations

import hashlib
from typing import Protocol

import diskcache


class Embedder(Protocol):
    """Protocol for document embedding models."""

    @property
    def model_name(self) -> str:
        ...

    @property
    def dimensions(self) -> int:
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class CachedEmbedder(Embedder):
    """Wraps an embedder with a diskcache to save API costs/time."""

    def __init__(self, embedder: Embedder, cache_dir: str = ".cache/embeddings"):
        self.embedder = embedder
        self._model_name = embedder.model_name
        self._dimensions = embedder.dimensions
        self.cache = diskcache.Cache(cache_dir)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for i, text in enumerate(texts):
            key = f"{self._model_name}::{self._hash(text)}"
            cached = self.cache.get(key)
            if cached is not None:
                results[i] = cached  # type: ignore
            else:
                missing_indices.append(i)
                missing_texts.append(text)

        if missing_texts:
            new_embeddings = self.embedder.embed_documents(missing_texts)
            for idx, text, emb in zip(missing_indices, missing_texts, new_embeddings):
                results[idx] = emb
                key = f"{self._model_name}::{self._hash(text)}"
                self.cache.set(key, emb)

        return results  # type: ignore

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
