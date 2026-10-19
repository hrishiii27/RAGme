"""Tests for the search modules."""

import pytest
from rag.types import Chunk
from rag.search.lexical import BM25Retriever
from rag.search.hybrid import rrf_combine
from rag.search.mmr import mmr_rerank

def test_rrf_combine():
    c1 = Chunk(chunk_id="c1", doc_id="d1", text="A", start_char=0, end_char=1, metadata={})
    c2 = Chunk(chunk_id="c2", doc_id="d2", text="B", start_char=0, end_char=1, metadata={})
    c3 = Chunk(chunk_id="c3", doc_id="d3", text="C", start_char=0, end_char=1, metadata={})
    
    lexical = [(c1, 10.0), (c2, 5.0), (c3, 1.0)]
    semantic = [(c3, 0.9), (c1, 0.8), (c2, 0.7)]
    
    results = rrf_combine(lexical, semantic, k=2, rank_constant=1)
    
    assert len(results) == 2
    assert results[0][0].chunk_id == "c1"  # Rank 1 + Rank 2
    assert results[1][0].chunk_id == "c3"  # Rank 3 + Rank 1

def test_mmr_rerank():
    # Embeddings for: "cat", "dog", "car", "truck"
    # Query is "pet" (closer to cat/dog)
    q = [1.0, 0.0]
    
    candidates = [
        [0.9, 0.1],  # cat (very close to query)
        [0.8, 0.2],  # dog (close to query, but very similar to cat)
        [0.0, 1.0],  # car (far from query)
    ]
    indices = [0, 1, 2]
    
    # If lambda=1.0 (only relevance), order is cat, dog, car
    res_rel = mmr_rerank(q, candidates, indices, k=2, lambda_param=1.0)
    assert res_rel == [0, 1]
    
    # If lambda=0.5, 'dog' gets penalized for being too similar to 'cat', so 'car' might win out for diversity
    res_div = mmr_rerank(q, candidates, indices, k=2, lambda_param=0.0)
    assert res_div == [0, 2]
