"""Pinecone Cloud Serverless Vector Store Implementation."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from rag.store.base import BaseVectorStore
from rag.types import Chunk

logger = logging.getLogger(__name__)


class PineconeStore(BaseVectorStore):
    """Production vector store powered by Pinecone Serverless (Free 2GB tier)."""

    def __init__(
        self,
        index_name: str = "bharat-rag",
        api_key: str | None = None,
        dimension: int = 384,
        cloud: str = "aws",
        region: str = "us-east-1",
    ):
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as e:
            raise ImportError(
                "pinecone package is required for PineconeStore. Install with: uv add pinecone"
            ) from e

        key = api_key or os.getenv("PINECONE_API_KEY")
        if not key:
            raise ValueError(
                "PINECONE_API_KEY is missing. Please set PINECONE_API_KEY in your .env file."
            )

        self.pc = Pinecone(api_key=key)
        self.index_name = index_name
        self.dimension = dimension

        # Verify or create serverless index
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            logger.info(
                f"Creating Pinecone Serverless index '{self.index_name}' "
                f"(dim={dimension}, metric=cosine, cloud={cloud}, region={region})..."
            )
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
            # Short grace period for serverless provisioning
            time.sleep(2)

        self.index = self.pc.Index(self.index_name)

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]], batch_size: int = 200) -> None:
        """Upsert chunks and embeddings in batches to Pinecone."""
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatched chunks ({len(chunks)}) and embeddings ({len(embeddings)})")

        if not chunks:
            return

        total = len(chunks)
        for i in range(0, total, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            vectors = []
            for c, emb in zip(batch_chunks, batch_embeddings):
                # Clean metadata (Pinecone supports strings, numbers, booleans, and list of strings)
                meta: dict[str, Any] = {
                    "doc_id": c.doc_id,
                    "text": c.text[:40000],  # Pinecone metadata size safeguard
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                }
                for k, v in c.metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        meta[k] = v
                    elif isinstance(v, list) and all(isinstance(x, str) for x in v):
                        meta[k] = v

                vectors.append({
                    "id": c.chunk_id,
                    "values": emb,
                    "metadata": meta,
                })

            self.index.upsert(vectors=vectors)
            logger.debug(f"Upserted batch {i}..{min(i + batch_size, total)} of {total} to Pinecone")

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Query Pinecone for top-k matching chunks."""
        response = self.index.query(
            vector=query_embedding,
            top_k=n_results,
            include_metadata=True,
            filter=where,
        )

        matches = response.get("matches", [])
        results = []
        for match in matches:
            meta = match.get("metadata", {})
            score = float(match.get("score") or 0.0)
            results.append({
                "id": match.get("id"),
                "document": meta.get("text"),
                "metadata": meta,
                "distance": 1.0 - score,  # Cosine distance
            })

        return results

    def count(self) -> int:
        """Return total vector count in index."""
        stats = self.index.describe_index_stats()
        return stats.total_vector_count or 0
