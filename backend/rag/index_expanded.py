"""Index expanded 15,000-document corpus into ChromaDB."""

from __future__ import annotations

import time
from pathlib import Path
import pyarrow.parquet as pq

from rag.chunking.strategies import RecursiveCharacterChunker
from rag.embed.base import CachedEmbedder
from rag.embed.local import LocalEmbedder
from rag.store.chroma import ChromaStore
from rag.types import Document, Chunk


def find_project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "data").exists() and (parent / "reports").exists():
            return parent
    return p.parents[3]


def main():
    root = find_project_root()
    parquet_path = root / "data" / "subset" / "english.parquet"
    collection_name = "A15000-recursive-1000-100"

    print(f"Loading 15,000-document corpus from {parquet_path}...")
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    print(f"Loaded {len(rows)} documents.")

    docs = [
        Document(
            doc_id=r["doc_id"],
            title=r.get("title") or "Bharat Government Document",
            text=r.get("text") or "",
            url=r.get("url") or "",
            metadata={
                "kind": r.get("kind", ""),
                "trust": r.get("trust", ""),
                "topics": r.get("topics", []),
            },
        )
        for r in rows
    ]

    print("Chunking documents with RecursiveCharacterChunker(1000, 100)...")
    chunker = RecursiveCharacterChunker(1000, 100)
    all_chunks: list[Chunk] = []
    for d in docs:
        all_chunks.extend(chunker.chunk(d))
    print(f"Total chunks produced: {len(all_chunks):,}")

    embedder = CachedEmbedder(
        LocalEmbedder("all-MiniLM-L6-v2"),
        cache_dir=str(root / ".cache" / "embeddings"),
    )
    store = ChromaStore(
        persist_dir=str(root / ".chroma"),
        collection_name=collection_name,
    )

    # Check if collection already has chunks
    existing_count = store.collection.count()
    print(f"Current collection count: {existing_count}")
    if existing_count >= len(all_chunks):
        print(f"Collection {collection_name} already contains {existing_count} chunks. Done!")
        return

    print("Embedding and indexing chunks into Chroma in batches...")
    batch_size = 1000
    t0 = time.time()

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        vecs = embedder.embed_documents(texts)
        store.add_chunks(batch, vecs)
        elapsed = time.time() - t0
        pct = (i + len(batch)) / len(all_chunks) * 100
        print(f"  [{pct:5.1f}%] Indexed {i + len(batch):,}/{len(all_chunks):,} chunks ({elapsed:.1f}s)")

    print(f"Successfully indexed {len(all_chunks):,} chunks into {collection_name} in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
