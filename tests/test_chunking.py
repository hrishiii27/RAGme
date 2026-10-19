"""Tests for the chunking module."""

import pytest
from rag.types import Document
from rag.chunking.strategies import FixedTokenChunker, RecursiveCharacterChunker, MarkdownHeaderChunker

def test_fixed_token_chunker():
    chunker = FixedTokenChunker(chunk_size=50, overlap=10)
    # 50 words is probably ~50-60 tokens
    text = "word " * 100
    doc = Document(doc_id="d1", title="test", text=text, url="", metadata={})
    
    chunks = chunker.chunk(doc)
    
    assert len(chunks) > 1
    assert chunks[0].doc_id == "d1"
    assert "word" in chunks[0].text
    
    # Overlap test roughly
    assert chunks[0].end_char > chunks[1].start_char
    assert text[chunks[0].start_char:chunks[0].end_char] == chunks[0].text

def test_fixed_token_chunker_overlap_too_large():
    with pytest.raises(ValueError):
        FixedTokenChunker(chunk_size=50, overlap=50)

def test_recursive_character_chunker():
    chunker = RecursiveCharacterChunker(chunk_size=50, overlap=10)
    text = "First paragraph.\n\nSecond paragraph which is a little longer to force a split.\n\nThird paragraph."
    doc = Document(doc_id="d2", title="test", text=text, url="", metadata={})
    
    chunks = chunker.chunk(doc)
    
    assert len(chunks) >= 3
    for chunk in chunks:
        assert len(chunk.text) <= 50
        assert text[chunk.start_char:chunk.end_char] == chunk.text

def test_markdown_header_chunker():
    chunker = MarkdownHeaderChunker()
    text = "# Title\nIntro text.\n## Section 1\nSection 1 text.\n### Subsection\nMore text.\n## Section 2\nSection 2 text."
    doc = Document(doc_id="d3", title="test", text=text, url="", metadata={})
    
    chunks = chunker.chunk(doc)
    
    assert len(chunks) == 4
    assert chunks[0].text.strip() == "# Title\nIntro text."
    assert chunks[1].text.strip() == "# Title > ## Section 1\nSection 1 text."
    assert chunks[2].text.strip() == "# Title > ## Section 1 > ### Subsection\nMore text."
    assert chunks[3].text.strip() == "# Title > ## Section 2\nSection 2 text."
