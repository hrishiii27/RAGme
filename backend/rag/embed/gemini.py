"""Phase 3 — Gemini API Embeddings."""

from __future__ import annotations

import os

from rag.embed.base import Embedder


class GeminiEmbedder(Embedder):
    """Google Gemini API Embedder.
    
    Requires GEMINI_API_KEY environment variable.
    """

    def __init__(
        self,
        model_name: str = "gemini-embedding-2",
        dimensions: int | None = None,
    ):
        self._model_name = model_name
        self._dimensions_val = dimensions
        
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "Could not import google.genai. "
                "Please install it with `uv add google-genai`."
            )

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable must be set.")
        
        self.client = genai.Client(api_key=api_key)
        self.types = types

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions_val or 3072

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
            
        batch_size = 100
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = self.client.models.embed_content(
                model=self._model_name,
                contents=batch,
                config=self.types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=self._dimensions_val
                )
            )
            if result.embeddings:
                for emb in result.embeddings:
                    all_embeddings.append(emb.values)
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        result = self.client.models.embed_content(
            model=self._model_name,
            contents=text,
            config=self.types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self._dimensions_val
            )
        )
        if not result.embeddings:
            raise RuntimeError("Gemini API returned empty embeddings.")
        return result.embeddings[0].values  # type: ignore
