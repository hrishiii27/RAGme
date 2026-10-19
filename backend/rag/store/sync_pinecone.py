"""Sync precomputed document chunks and embeddings to Pinecone Cloud."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from rag.pipeline import find_project_root, RAGPipeline
from rag.store.pinecone import PineconeStore
from rag.types import Document
import pyarrow.parquet as pq


def sync_to_pinecone(batch_size: int = 200, max_chunks: int | None = None) -> None:
    root = find_project_root()
    parquet_path = root / "data" / "subset" / "english.parquet"

    print(f"Loading documents from {parquet_path}...")
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()

    docs = [
        Document(
            doc_id=r["doc_id"],
            title=r.get("title") or "Bharat Government Document",
            text=r.get("text") or "",
            url=r.get("url") or "",
            metadata={
                "kind": r.get("kind", ""),
                "topics": r.get("topics", []),
                "trust": r.get("trust", ""),
            },
        )
        for r in rows
    ]

    print(f"Initializing RAG Pipeline over {len(docs)} documents...")
    # Initialize pipeline with chroma temporary fallback to generate/load chunks
    pipeline = RAGPipeline(docs=docs, store_type="chroma")
    chunks = pipeline.all_chunks

    if max_chunks:
        chunks = chunks[:max_chunks]
        print(f"Limiting upload to first {max_chunks} chunks.")

    print(f"Total chunks to sync: {len(chunks)}")

    # Initialize Pinecone Store
    index_name = os.getenv("PINECONE_INDEX_NAME", "bharat-rag")
    pinecone_store = PineconeStore(index_name=index_name)
    current_count = pinecone_store.count()
    print(f"Current vectors in Pinecone index '{index_name}': {current_count}")

    if current_count >= len(chunks):
        print("Pinecone index already contains all chunks! Nothing to sync.")
        return

    print("Embedding and uploading chunks to Pinecone in batches...")
    for i in tqdm(range(0, len(chunks), batch_size), desc="Pinecone Upsert"):
        batch_chunks = chunks[i : i + batch_size]
        batch_texts = [c.text for c in batch_chunks]
        # Embed batch (CachedEmbedder retrieves from local disk cache instantaneously)
        batch_embeddings = pipeline.embedder.embed_documents(batch_texts)
        pinecone_store.add_chunks(batch_chunks, batch_embeddings, batch_size=batch_size)

    final_count = pinecone_store.count()
    print(f"\nSuccessfully synced! Final Pinecone vector count: {final_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync document chunks to Pinecone Cloud")
    parser.add_argument("--batch-size", type=int, default=200, help="Batch size for upserting")
    parser.add_argument("--limit", type=int, default=None, help="Max chunks to sync")
    args = parser.parse_args()

    sync_to_pinecone(batch_size=args.batch_size, max_chunks=args.limit)
