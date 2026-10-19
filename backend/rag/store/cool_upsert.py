"""Low-thermal, streaming upsert to Pinecone Serverless Cloud.

Engineered specifically for low memory footprint and cool CPU on Apple Silicon:
- Restricts PyTorch to 2 threads (prevents 100% CPU spikes / chassis overheating).
- Streams document shards lazily instead of loading 50k rows into RAM.
- Flushes garbage collection and sleeps briefly between batches.
- Skips documents already indexed in Pinecone.

Usage:
    uv run python -m rag.store.cool_upsert --max-docs 2500 --batch-docs 100
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Limit CPU threads before importing torch/sentence_transformers
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))
load_dotenv(ROOT / ".env")

import pyarrow.parquet as pq
import torch

torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

from sentence_transformers import SentenceTransformer
from rag.chunking.strategies import RecursiveCharacterChunker
from rag.store.pinecone import PineconeStore
from rag.types import Chunk, Document


def stream_unindexed_documents(existing_ids: set[str], max_docs: int) -> list[Document]:
    """Stream raw documents from parquet shards that are not yet indexed."""
    raw_shards = sorted((ROOT / "data" / "raw" / "data").glob("*.parquet"))
    found_docs: list[Document] = []
    
    cols = ["document_id", "title", "text", "url", "kind", "trust", "topics"]
    
    for shard in raw_shards:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=500, columns=cols):
            for row in batch.to_pylist():
                doc_id = row["document_id"]
                if doc_id in existing_ids:
                    continue
                
                text = (row.get("text") or "").strip()
                if len(text) < 400:  # Skip tiny stubs / empty errors
                    continue
                
                doc = Document(
                    doc_id=doc_id,
                    title=row.get("title") or "Bharat Government Document",
                    text=text,
                    url=row.get("url") or "",
                    metadata={
                        "kind": row.get("kind", ""),
                        "trust": row.get("trust", ""),
                        "topics": row.get("topics", []),
                    },
                )
                found_docs.append(doc)
                if len(found_docs) >= max_docs:
                    return found_docs
    return found_docs


def run_cool_upsert(
    max_docs: int = 0,
    target_vectors: int = 500000,
    batch_docs: int = 100,
    cool_down_sec: float = 0.25,
) -> None:
    print("=" * 65)
    print("🌿 BharatVault Cool-Mode Pinecone Cloud Upsert")
    print(f"Target Total Vectors: {target_vectors:,} | Batch size: {batch_docs} docs")
    print("Thermal limit: 2 CPU threads | Thermal pause between batches")
    print("=" * 65)

    # 1. Connect to Pinecone and check current vector count
    pinecone_store = PineconeStore()
    initial_vector_count = pinecone_store.count()
    print(f"Current Pinecone Cloud vector count: {initial_vector_count:,}")

    if initial_vector_count >= target_vectors:
        print(f"✅ Target of {target_vectors:,} vectors already reached! No indexing required.")
        return

    vectors_needed = target_vectors - initial_vector_count
    estimated_docs_needed = int(vectors_needed / 14.59) + 20
    docs_to_fetch = min(max_docs, estimated_docs_needed) if max_docs > 0 else estimated_docs_needed
    print(f"Vectors needed: {vectors_needed:,} (approx. {docs_to_fetch:,} new documents)")

    # 2. Load known existing doc_ids from english.parquet and tracker file
    tracker_path = ROOT / "data" / "subset" / "indexed_doc_ids.txt"
    en_path = ROOT / "data" / "subset" / "english.parquet"
    existing_ids: set[str] = set()

    if en_path.exists():
        existing_en = pq.read_table(en_path, columns=["doc_id"]).to_pandas()
        existing_ids.update(existing_en["doc_id"])
    if tracker_path.exists():
        with open(tracker_path, "r") as f:
            existing_ids.update(line.strip() for line in f if line.strip())

    print(f"Existing indexed doc_ids across all sources: {len(existing_ids):,}")

    # 3. Gather new candidate documents
    print(f"Scanning raw shards for {docs_to_fetch:,} unindexed official documents...")
    t0 = time.time()
    new_docs = stream_unindexed_documents(existing_ids, max_docs=docs_to_fetch)
    print(f"Found {len(new_docs):,} new candidate documents in {time.time() - t0:.1f}s.")

    if not new_docs:
        print("No new documents to index. Pinecone is completely up to date.")
        return

    # 4. Initialize chunker and embedding model
    chunker = RecursiveCharacterChunker(1000, 100)
    print("Loading SentenceTransformer ('all-MiniLM-L6-v2') in low-thread mode...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 5. Stream in batches to keep memory < 150MB and CPU cool
    total_chunks_upserted = 0
    total_docs = len(new_docs)

    for i in range(0, total_docs, batch_docs):
        batch = new_docs[i : i + batch_docs]
        
        # Chunk docs
        batch_chunks: list[Chunk] = []
        for d in batch:
            batch_chunks.extend(chunker.chunk(d))

        if not batch_chunks:
            continue

        # Embed batch
        texts = [c.text for c in batch_chunks]
        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()

        # Upsert directly to Pinecone
        pinecone_store.add_chunks(batch_chunks, embeddings, batch_size=100)
        total_chunks_upserted += len(batch_chunks)

        # Record doc_ids to tracker file
        with open(tracker_path, "a") as f:
            for d in batch:
                f.write(f"{d.doc_id}\n")

        current_total = initial_vector_count + total_chunks_upserted
        pct = min(100.0, (current_total / target_vectors) * 100)
        print(
            f"  [Progress {pct:5.1f}%] Processed {i + len(batch):,}/{total_docs:,} docs "
            f"→ +{len(batch_chunks):,} chunks | Pinecone Total: {current_total:,} / {target_vectors:,}"
        )

        # Stop if target reached
        if current_total >= target_vectors:
            print(f"🎯 Target reached: {current_total:,} >= {target_vectors:,} vectors!")
            break

        # Thermal breathing room & memory reclamation
        del batch_chunks
        del texts
        del embeddings
        gc.collect()
        time.sleep(cool_down_sec)

    final_count = pinecone_store.count()
    print("=" * 65)
    print(f"✅ Upsert complete! Added {total_chunks_upserted:,} new vectors to Pinecone.")
    print(f"📈 Total Pinecone Cloud Vectors: {initial_vector_count:,} → {final_count:,}")
    print("=" * 65)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-docs", type=int, default=0, help="Max documents to index (0 = auto-compute for target)")
    ap.add_argument("--target-vectors", type=int, default=500000, help="Target total vectors in Pinecone")
    ap.add_argument("--batch-docs", type=int, default=100, help="Documents per batch")
    ap.add_argument("--cool-down", type=float, default=0.25, help="Seconds to sleep between batches")
    args = ap.parse_args()

    run_cool_upsert(
        max_docs=args.max_docs,
        target_vectors=args.target_vectors,
        batch_docs=args.batch_docs,
        cool_down_sec=args.cool_down,
    )
