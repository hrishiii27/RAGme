"""Phase 3 — Vector Store using ChromaDB (Supports Local Embedded & Containerized HTTP Client)."""

from __future__ import annotations

import logging
import os
from typing import Any

import chromadb
from chromadb.api.types import Documents, Embeddings, Metadatas

from rag.store.base import BaseVectorStore
from rag.types import Chunk

logger = logging.getLogger(__name__)


class ChromaStore(BaseVectorStore):
    """Vector store using ChromaDB.
    
    Supports:
    1. Local Embedded Mode (default): uses SQLite on local filesystem
    2. Container/HTTP Mode: uses chromadb.HttpClient when host/port is provided or CHROMA_HOST is set
    """

    def __init__(
        self,
        persist_dir: str = ".chroma",
        collection_name: str = "rag_default",
        host: str | None = None,
        port: int | None = None,
    ):
        chroma_host = host or os.getenv("CHROMA_HOST")
        chroma_port_env = os.getenv("CHROMA_PORT")
        chroma_port = port or (int(chroma_port_env) if chroma_port_env else 8000)

        if chroma_host:
            logger.info(f"Connecting to containerized/remote ChromaDB at {chroma_host}:{chroma_port}")
            self.client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        else:
            self.client = chromadb.PersistentClient(path=persist_dir)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,  # Dense embeddings precomputed via sentence-transformers
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Add chunks and their pre-computed embeddings to the store."""
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatched chunks ({len(chunks)}) and embeddings ({len(embeddings)})")

        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        metadatas: list[dict[str, str | int | float | bool]] = []

        for c in chunks:
            # Chroma doesn't allow complex nested objects in metadata
            meta: dict[str, str | int | float | bool] = {
                "doc_id": c.doc_id,
                "start_char": c.start_char,
                "end_char": c.end_char,
            }
            for k, v in c.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                elif isinstance(v, list) and all(isinstance(i, str) for i in v):
                    meta[k] = ",".join(v)
            metadatas.append(meta)

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self, query_embedding: list[float], n_results: int = 5, where: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Search the store and return raw results."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
        )

        out = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                out.append({
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i] if results["documents"] else None,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else None,
                    "distance": results["distances"][0][i] if results["distances"] else None,
                })
        return out

    def count(self) -> int:
        """Return total vectors in the collection."""
        return self.collection.count()
