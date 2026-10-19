"""Phase 1c — Subset selection (stratified sampling)."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from rag.config import settings
from rag.ingest.clean import clean_document
from rag.ingest.load import iter_raw_documents
from rag.types import Document


def select_subset(n_english: int = 6000, n_indic: int = 500) -> None:
    settings.subset_dir.mkdir(parents=True, exist_ok=True)
    
    # We want a mix of topics and kinds, but the dataset is heavily skewed towards
    # "page" (68%) and "pdf" (24%). A simple top-N per topic would pull mostly pages.
    # For now, we do a simple pass: keep docs < 100k chars, bin them into (topic, kind, lang),
    # and pull evenly from the bins until we hit our targets.
    
    en_docs: list[Document] = []
    indic_docs: list[Document] = []
    
    # Group by (primary_topic, kind) -> list of docs
    # This is a naive online reservoir sampling without the random part, just filling buckets.
    # We'll just stream and keep up to `limit_per_bucket` per bucket to force diversity.
    buckets_en: dict[tuple[str, str], list[Document]] = {}
    buckets_indic: dict[tuple[str, str], list[Document]] = {}
    
    print("Scanning and cleaning documents...")
    limit_per_bucket = max(150, n_english // 35)  # Dynamic bucket scaling
    
    for i, raw_doc in enumerate(iter_raw_documents()):
        if i % 5000 == 0 and i > 0:
            print(f"  Scanned {i}... (English: {len(en_docs)}/{n_english}, Indic: {len(indic_docs)}/{n_indic})")
            
        if len(en_docs) >= n_english and len(indic_docs) >= n_indic:
            break
            
        if raw_doc.metadata["n_chars"] > 100_000:
            continue  # Drop giants (per B2 decision)
            
        clean_doc = clean_document(raw_doc)
        if not clean_doc.text.strip() or clean_doc.metadata["ocr_score"] > 0.4:
            continue  # Drop empty or heavy garbage
            
        primary_topic = clean_doc.metadata["topics"][0] if clean_doc.metadata["topics"] else "unclassified"
        kind = clean_doc.metadata["kind"]
        lang = clean_doc.metadata["lang"]
        
        key = (primary_topic, kind)
        
        if lang == "en" and len(en_docs) < n_english:
            b = buckets_en.setdefault(key, [])
            if len(b) < limit_per_bucket:
                b.append(clean_doc)
                en_docs.append(clean_doc)
                
        elif lang == "indic" and len(indic_docs) < n_indic:
            b = buckets_indic.setdefault(key, [])
            if len(b) < limit_per_bucket:
                b.append(clean_doc)
                indic_docs.append(clean_doc)
                
    # Save the subsets
    print(f"Writing {len(en_docs)} English docs and {len(indic_docs)} Indic docs...")
    _write_parquet(en_docs, settings.subset_dir / "english.parquet")
    _write_parquet(indic_docs, settings.subset_dir / "indic.parquet")
    print(f"Subset selection complete! Generated {len(en_docs)} English docs in {settings.subset_dir / 'english.parquet'}.")


def _write_parquet(docs: list[Document], path: str | pa.lib.StringScalar) -> None:
    if not docs:
        return
        
    # Flatten the metadata back out for Parquet
    rows = []
    for d in docs:
        row = {
            "doc_id": d.doc_id,
            "title": d.title,
            "text": d.text,
            "url": d.url,
            **d.metadata
        }
        rows.append(row)
        
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--english", type=int, default=15000, help="Number of English documents to sample")
    parser.add_argument("--indic", type=int, default=1000, help="Number of Indic documents to sample")
    args = parser.parse_args()
    select_subset(n_english=args.english, n_indic=args.indic)
