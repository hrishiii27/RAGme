"""RAGme core library: every RAG phase lives in its own subpackage.

ingest/     -> download, load, clean, subset
chunking/   -> Chunker implementations
embedding/  -> Embedder implementations      (next)
index/      -> Chroma + BM25 stores          (next)
retrieval/  -> dense / sparse / hybrid       (next)
generation/ -> LLM clients + prompts         (Day 2)
"""
