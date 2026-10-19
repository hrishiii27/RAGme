"""Tests for Anthropic-Style Contextual Chunking."""

import pytest
from rag.chunking.contextual import ContextualChunker
from rag.chunking.strategies import RecursiveCharacterChunker
from rag.types import Document


def test_contextual_chunker_heuristic(tmp_path):
    doc = Document(
        doc_id="doc_cmvr_01",
        title="Central Motor Vehicles Rules 1989",
        text="Sub-rule (3): The fee payable for an application under clause (b) shall be three hundred rupees.",
        url="https://morth.nic.in/cmvr-rules",
        metadata={
            "host": "morth.nic.in",
            "topics": ["transport", "rules"],
            "kind": "statute",
        },
    )

    chunker = ContextualChunker(
        base_chunker=RecursiveCharacterChunker(500, 50),
        mode="heuristic",
        cache_dir=tmp_path,
    )

    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.doc_id == "doc_cmvr_01"
    assert "Document: Central Motor Vehicles Rules 1989" in chunk.text
    assert "Portal: morth.nic.in" in chunk.text
    assert "Sub-rule (3)" in chunk.text
    assert chunk.metadata["raw_text"] == doc.text


def test_contextual_chunker_llm_mode(tmp_path):
    class MockLLM:
        def generate(self, prompt: str) -> str:
            return "This section outlines the application fee requirement under Motor Vehicles regulations."

    doc = Document(
        doc_id="doc_fee_02",
        title="Fee Guidelines",
        text="The fee payable shall be 500 rupees.",
        url="https://gov.in",
        metadata={},
    )

    chunker = ContextualChunker(
        base_chunker=RecursiveCharacterChunker(500, 50),
        mode="llm",
        cache_dir=tmp_path,
        llm=MockLLM(),
    )

    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert "This section outlines the application fee requirement" in chunk.text
    assert "500 rupees" in chunk.text
    assert chunk.metadata["context_mode"] == "llm"
