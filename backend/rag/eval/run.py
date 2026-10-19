"""Phase 5 — Experiment runner across stages:
Stage A: Chunking sweep (Dense retrieval)
Stage C: Search sweep (BM25 vs Dense vs Hybrid RRF vs Hybrid + MMR) using the winning chunker (recursive-1000-100)

Usage:
    uv run python -m rag.eval.run --stage C --limit-docs 6000
    uv run python -m rag.eval.run --stage C --only-reviewed
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from rag.chunking.strategies import FixedTokenChunker, RecursiveCharacterChunker
from rag.embed.base import CachedEmbedder
from rag.embed.local import LocalEmbedder
from rag.eval.metrics import hit_at_k, mrr, ndcg_at_k, recall_at_k
from rag.search.hybrid import rrf_combine
from rag.search.lexical import BM25Retriever
from rag.search.mmr import mmr_rerank
from rag.store.chroma import ChromaStore
from rag.types import Chunk, Document

SUBSET = Path("data/subset/english.parquet")
GOLDEN = Path("data/golden/synthetic_qa.jsonl")
REPORTS = Path("reports")
EMBED_BATCH = 128
K = 10


def load_docs(limit: int | None, must_include: set[str]) -> list[Document]:
    """Load the subset, always keeping docs referenced by the golden set so labels stay valid."""
    rows = pq.read_table(SUBSET).to_pylist()
    docs = [
        Document(doc_id=r["doc_id"], title=r["title"], text=r["text"], url=r["url"], metadata={})
        for r in rows
    ]
    if limit and limit < len(docs):
        keep = [d for d in docs if d.doc_id in must_include]
        rest = [d for d in docs if d.doc_id not in must_include]
        docs = keep + rest[: max(0, limit - len(keep))]
    return docs


def load_golden() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text().splitlines() if line]


def stage_a_configs() -> dict:
    return {
        "fixed-256-32": FixedTokenChunker(256, 32),
        "fixed-512-64": FixedTokenChunker(512, 64),
        "recursive-1000-100": RecursiveCharacterChunker(1000, 100),
        "recursive-1500-150": RecursiveCharacterChunker(1500, 150),
    }


def build_index(name: str, chunker, docs: list[Document], embedder: CachedEmbedder,
                tag: str) -> tuple[ChromaStore, list[Chunk], int, float]:
    store = ChromaStore(persist_dir=".chroma", collection_name=f"{tag}-{name}")
    
    # We collect all chunks in memory so BM25 can also index them in Stage C
    all_chunks: list[Chunk] = []
    for d in docs:
        all_chunks.extend(chunker.chunk(d))
        
    existing = store.collection.count()
    if existing:
        print(f"  [{name}] reusing {existing} indexed chunks in Chroma")
        return store, all_chunks, existing, 0.0

    t0 = time.time()
    buf: list[Chunk] = []
    total = 0

    def flush() -> None:
        nonlocal total
        if not buf:
            return
        embs = embedder.embed_documents([c.text for c in buf])
        store.add_chunks(buf, embs)
        total += len(buf)
        buf.clear()

    for i, c in enumerate(all_chunks):
        buf.append(c)
        if len(buf) >= EMBED_BATCH * 4:
            flush()
    flush()
    return store, all_chunks, total, time.time() - t0


def evaluate_retrieval(query_fn, golden: list[dict]) -> dict:
    agg: dict[str, float] = defaultdict(float)
    for g in golden:
        retrieved_doc_ids = query_fn(g["question"])
        rel = set(g["relevant_doc_ids"])
        agg["recall@5"] += recall_at_k(retrieved_doc_ids, rel, 5)
        agg["recall@10"] += recall_at_k(retrieved_doc_ids, rel, 10)
        agg["hit@1"] += hit_at_k(retrieved_doc_ids, rel, 1)
        agg["mrr"] += mrr(retrieved_doc_ids, rel)
        agg["ndcg@10"] += ndcg_at_k(retrieved_doc_ids, rel, 10)
    n = len(golden)
    return {k: round(v / n, 4) for k, v in agg.items()}


def run_stage_a(docs: list[Document], golden: list[dict]) -> list[dict]:
    embedder = CachedEmbedder(LocalEmbedder("all-MiniLM-L6-v2"), cache_dir=".cache/embeddings")
    tag = f"A{len(docs)}"
    rows = []
    for name, chunker in stage_a_configs().items():
        print(f"== Stage A: {name}")
        store, _, n_chunks, secs = build_index(name, chunker, docs, embedder, tag)
        def query_fn(q_text: str):
            q_vec = embedder.embed_query(q_text)
            hits = store.search(q_vec, n_results=K * 3)
            return [h["metadata"]["doc_id"] for h in hits]
        metrics = evaluate_retrieval(query_fn, golden)
        row = {"config": name, "n_docs": len(docs), "n_chunks": n_chunks,
               "index_secs": round(secs), **metrics}
        print("  ", row)
        rows.append(row)
    return rows


def run_stage_c(docs: list[Document], golden: list[dict]) -> list[dict]:
    embedder = CachedEmbedder(LocalEmbedder("all-MiniLM-L6-v2"), cache_dir=".cache/embeddings")
    tag = f"A{len(docs)}"
    chunker = RecursiveCharacterChunker(1000, 100)
    name = "recursive-1000-100"
    
    print(f"Loading/preparing index for Stage C with winner: {name}...")
    store, all_chunks, n_chunks, _ = build_index(name, chunker, docs, embedder, tag)
    
    print(f"Building BM25 index over {len(all_chunks)} chunks...")
    t0 = time.time()
    bm25 = BM25Retriever(all_chunks)
    bm25_sec = round(time.time() - t0, 1)
    print(f"BM25 index ready in {bm25_sec}s.")
    
    # Map chunk_id to Chunk for lookups
    chunk_by_id: dict[str, Chunk] = {c.chunk_id: c for c in all_chunks}

    search_modes = ["dense-only", "bm25-only", "hybrid-rrf", "hybrid-rrf-mmr"]
    rows = []

    for mode in search_modes:
        print(f"== Stage C: {mode}")
        t_start = time.time()
        
        def query_fn(q_text: str, m=mode):
            q_vec = embedder.embed_query(q_text) if "dense" in m or "hybrid" in m else None
            
            # Dense candidate fetch
            dense_hits = []
            if q_vec is not None:
                d_results = store.search(q_vec, n_results=30)
                for r in d_results:
                    cid = r["id"]
                    c = chunk_by_id.get(cid)
                    if c:
                        # Chroma distance to cosine/similarity approx
                        dense_hits.append((c, 1.0 - (r["distance"] or 0.0)))
            
            # BM25 candidate fetch
            bm25_hits = []
            if "bm25" in m or "hybrid" in m:
                bm25_hits = bm25.search(q_text, k=30)
                
            if m == "dense-only":
                return [c.doc_id for c, _ in dense_hits]
            elif m == "bm25-only":
                return [c.doc_id for c, _ in bm25_hits]
            elif m == "hybrid-rrf":
                fused = rrf_combine(bm25_hits, dense_hits, k=25, rank_constant=60)
                return [c.doc_id for c, _ in fused]
            elif m == "hybrid-rrf-mmr":
                fused = rrf_combine(bm25_hits, dense_hits, k=25, rank_constant=60)
                if not fused:
                    return []
                # Candidate embeddings for MMR
                top_chunks = [c for c, _ in fused[:15]]
                cand_embs = embedder.embed_documents([c.text for c in top_chunks])
                indices = list(range(len(top_chunks)))
                reranked_idx = mmr_rerank(q_vec, cand_embs, indices, k=K * 2, lambda_param=0.7)
                return [top_chunks[i].doc_id for i in reranked_idx]

        metrics = evaluate_retrieval(query_fn, golden)
        elapsed = round(time.time() - t_start, 1)
        row = {"config": mode, "search_time_s": elapsed, **metrics}
        print("  ", row)
        rows.append(row)

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="C", choices=["A", "C"])
    ap.add_argument("--limit-docs", type=int, default=6000)
    ap.add_argument("--only-reviewed", action="store_true", help="score only hand-reviewed Qs")
    args = ap.parse_args()

    golden = load_golden()
    if args.only_reviewed:
        golden = [g for g in golden if g.get("reviewed")]
    must = {d for g in golden for d in g["relevant_doc_ids"]}
    docs = load_docs(args.limit_docs, must)
    print(f"Ready: {len(docs)} docs, {len(golden)} questions (only_reviewed={args.only_reviewed})")

    if args.stage == "A":
        rows = run_stage_a(docs, golden)
    else:
        rows = run_stage_c(docs, golden)

    REPORTS.mkdir(exist_ok=True)
    report_file = REPORTS / f"stage_{args.stage}.json"
    report_file.write_text(json.dumps(rows, indent=2))
    
    cols = list(rows[0].keys())
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    md_str = "\n".join(md) + "\n"
    (REPORTS / f"stage_{args.stage}.md").write_text(md_str)
    print("\n" + md_str)


if __name__ == "__main__":
    main()
