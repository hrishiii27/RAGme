"""Phase 3 — Local Embeddings via sentence-transformers."""

from __future__ import annotations

import logging

from rag.embed.base import Embedder

logger = logging.getLogger(__name__)


class LocalEmbedder(Embedder):
    """Sentence-transformers based local embedder."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Could not import sentence_transformers. "
                "Please install it with `uv add sentence-transformers`."
            )
        logger.info(f"Loading local model {model_name}...")
        self.model = SentenceTransformer(model_name)
        if hasattr(self.model, "get_embedding_dimension"):
            dim = self.model.get_embedding_dimension()
        else:
            dim = self.model.get_sentence_embedding_dimension()
        self._dimensions = int(dim) if dim is not None else 384

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()  # type: ignore

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
