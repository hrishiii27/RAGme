"""Qdrant Cloud & Cluster Vector Store Implementation."""

from __future__ import annotations

import logging
from typing import Any

from rag.store.base import BaseVectorStore
from rag.types import Chunk

logger = logging.getLogger(__name__)


class QdrantStore(BaseVectorStore):
    """Cloud vector store powered by Qdrant (supports Qdrant Cloud Free Tier & Docker)."""

    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        collection_name: str = "rag_default",
        vector_dim: int = 384,
    ):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as rest
        except ImportError as e:
            raise ImportError(
                "qdrant-client is required to use QdrantStore. Install with: uv add qdrant-client"
            ) from e

        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self.vector_dim = vector_dim

        # Ensure collection exists
        collections = [c.name for c in self.client.get_collections().collections]
        if collection_name not in collections:
            logger.info(f"Creating Qdrant collection '{collection_name}' (dim={vector_dim}, distance=Cosine)")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=rest.VectorParams(
                    size=vector_dim,
                    distance=rest.Distance.COSINE,
                ),
            )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        from qdrant_client.http import models as rest

        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatched chunks ({len(chunks)}) and embeddings ({len(embeddings)})")

        if not chunks:
            return

        points = []
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            payload = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                **chunk.metadata,
            }
            points.append(
                rest.PointStruct(
                    id=idx,
                    vector=emb,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=n_results,
        )

        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append({
                "id": payload.get("chunk_id", str(hit.id)),
                "document": payload.get("text"),
                "metadata": payload,
                "distance": 1.0 - hit.score,  # Convert cosine similarity to distance
            })
        return results

    def count(self) -> int:
        info = self.client.get_collection(self.collection_name)
        return info.points_count or 0
