"""Vector Store Factory supporting Pinecone Cloud, Local Chroma, and Qdrant."""

from __future__ import annotations

import logging
import os
from typing import Any

from rag.store.base import BaseVectorStore
from rag.store.chroma import ChromaStore

logger = logging.getLogger(__name__)


def create_vector_store(
    collection_name: str = "rag_default",
    persist_dir: str = ".chroma",
    store_type: str | None = None,
    **kwargs: Any,
) -> BaseVectorStore:
    """Factory creating the appropriate vector store backend.
    
    Controlled by `VECTOR_STORE_TYPE` env variable (defaults to 'pinecone' if configured, else 'chroma'):
    - 'pinecone': Connects to Pinecone Serverless Cloud (2GB free tier)
    - 'chroma': Uses ChromaDB (Local SQLite file or containerized HttpClient if CHROMA_HOST is set)
    - 'qdrant': Connects to Qdrant Cloud or remote cluster via QDRANT_URL and QDRANT_API_KEY
    """
    default_type = "pinecone" if os.getenv("PINECONE_API_KEY") else "chroma"
    selected_type = (store_type or os.getenv("VECTOR_STORE_TYPE", default_type)).lower().strip()

    if selected_type == "pinecone":
        index_name = os.getenv("PINECONE_INDEX_NAME", "bharat-rag")
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError(
                "VECTOR_STORE_TYPE=pinecone requested but PINECONE_API_KEY environment variable is missing. "
                "Provide your Pinecone API key in .env."
            )
        try:
            from rag.store.pinecone import PineconeStore
            return PineconeStore(
                index_name=index_name,
                api_key=api_key,
                dimension=kwargs.get("dimension", 384),
            )
        except ImportError as e:
            raise ImportError(
                "pinecone package is required for Pinecone. Install with: uv add pinecone"
            ) from e

    elif selected_type == "chroma":
        return ChromaStore(
            persist_dir=persist_dir,
            collection_name=collection_name,
            host=kwargs.get("host") or os.getenv("CHROMA_HOST"),
            port=kwargs.get("port") or (int(os.getenv("CHROMA_PORT", "8000")) if os.getenv("CHROMA_PORT") else None),
        )

    elif selected_type == "qdrant":
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_key = os.getenv("QDRANT_API_KEY")
        if not qdrant_url:
            raise ValueError(
                "VECTOR_STORE_TYPE=qdrant requested but QDRANT_URL environment variable is missing. "
                "Provide your Qdrant Cloud cluster URL in .env."
            )
        try:
            from rag.store.qdrant import QdrantStore
            return QdrantStore(
                url=qdrant_url,
                api_key=qdrant_key,
                collection_name=collection_name,
            )
        except ImportError as e:
            raise ImportError(
                "qdrant-client package is required for Qdrant Cloud. Install with: uv add qdrant-client"
            ) from e

    else:
        raise ValueError(
            f"Unsupported vector store type '{selected_type}'. Supported backends: 'pinecone', 'chroma', 'qdrant'."
        )
