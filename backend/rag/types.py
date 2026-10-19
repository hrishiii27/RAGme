"""Shared data types that flow between pipeline phases.

    raw parquet row --(load)--> Document --(clean)--> Document --(chunk)--> list[Chunk]

Keeping these tiny and explicit is deliberate: every phase speaks the same language,
so any Chunker can feed any Embedder, any Retriever can feed any PromptBuilder, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    url: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # metadata keys we populate: host, kind, trust, quality, topics, fetched_at,
    #                            lang, ocr_score, n_chars


@dataclass
class Chunk:
    chunk_id: str          # f"{doc_id}::{strategy}::{index}" — unique per chunking strategy
    doc_id: str            # golden labels are DOC-level, so evals work across strategies
    text: str              # what gets embedded / BM25-indexed
    start_char: int        # span in the *cleaned* document text (enables span-level evals)
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)
    # parent-child chunking stores e.g. metadata["parent_text"] for small-to-big expansion
