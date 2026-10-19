"""Load raw Parquet rows into `Document` objects (no cleaning here — that's clean.py).

Memory note (8 GB laptop): the full text column is ~1 GB, so we stream record batches
with pyarrow instead of pandas.read_parquet on everything.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlparse

import pyarrow.parquet as pq

from rag.config import settings
from rag.types import Document

COLUMNS = ["document_id", "title", "text", "url", "kind", "trust", "quality",
           "fetched_at", "topics"]


def iter_raw_documents(raw_dir: Path | None = None, batch_size: int = 512) -> Iterator[Document]:
    raw_dir = raw_dir or settings.raw_dir
    shards = sorted((raw_dir / "data").glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards in {raw_dir}/data — run rag.ingest.download")
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=batch_size, columns=COLUMNS):
            for row in batch.to_pylist():
                yield row_to_document(row)


def row_to_document(row: dict) -> Document:
    return Document(
        doc_id=row["document_id"],
        title=row["title"] or "",
        text=row["text"] or "",
        url=row["url"] or "",
        metadata={
            "host": urlparse(row["url"] or "").netloc.lower(),
            "kind": row["kind"],
            "trust": row["trust"],
            "quality": row["quality"],
            "fetched_at": row["fetched_at"],
            "topics": list(row["topics"] or []),
            "n_chars": len(row["text"] or ""),
        },
    )
