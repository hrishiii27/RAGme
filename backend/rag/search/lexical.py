"""Phase 4 — Lexical Search (BM25). 🧑‍💻 YOUR ASSIGNMENT

Task: Implement BM25 search logic. 
You can use `rank_bm25` (install with `uv add rank-bm25`) or write your own.
"""

from __future__ import annotations
 
import math
import re
from collections import Counter
 
from rag.types import Chunk
 
# Word characters plus the whole Indic block (U+0900-U+0DFF). Python's \w drops
# combining vowel signs (matras), which would split Hindi/Punjabi words apart.
_TOKEN_RE = re.compile(r"[\wऀ-෿]+")
 
 
def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())
 
 
class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
 
        n = len(chunks)
        self.doc_len: list[int] = []
        # term -> list of (chunk position, term frequency in that chunk)
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, chunk in enumerate(chunks):
            tokens = tokenize(chunk.text)
            self.doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self.postings.setdefault(term, []).append((i, tf))
 
        self.avgdl = (sum(self.doc_len) / n) if n else 0.0
        # Lucene-style IDF: always >= 0, unlike the classic formula which goes
        # negative for terms that appear in more than half the chunks.
        self.idf = {
            term: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self.postings.items()
        }
 
    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        if k <= 0 or not self.chunks or self.avgdl == 0:
            return []
        scores: dict[int, float] = {}
        for term in set(tokenize(query)):  # repeated query words count once
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for i, tf in postings:
                norm = 1 - self.b + self.b * self.doc_len[i] / self.avgdl
                scores[i] = scores.get(i, 0.0) + idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
 
        # Highest score first; ties broken by original position for determinism.
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [(self.chunks[i], s) for i, s in ranked if s > 0]
 