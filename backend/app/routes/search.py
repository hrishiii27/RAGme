"""Search inspector endpoint for granular retrieval diagnostics."""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas import SearchRequest, SearchResponse, SourceChunk
from app.services import get_pipeline
from rag.search.hybrid import rrf_combine

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search_endpoint(request: SearchRequest):
    pipeline = get_pipeline()
    query = request.query
    top_k = request.top_k

    if request.strategy == "dense":
        q_vec = pipeline.embedder.embed_query(query)
        d_res = pipeline.store.search(q_vec, n_results=top_k)
        hits = []
        for r in d_res:
            c = pipeline.chunk_by_id.get(r["id"])
            if c:
                hits.append((c, 1.0 - (r["distance"] or 0.0)))
    elif request.strategy == "bm25":
        hits = pipeline.bm25.search(query, k=top_k)
    else:  # hybrid_rrf
        hits = pipeline.retrieve(query, top_k=top_k)

    results = [
        SourceChunk(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            title=c.metadata.get("title") or "Bharat Government Document",
            url=c.metadata.get("url") or "",
            score=round(float(score), 4),
            snippet=c.text[:300] + ("..." if len(c.text) > 300 else ""),
        )
        for c, score in hits
    ]

    return SearchResponse(
        query=query,
        count=len(results),
        strategy=request.strategy,
        results=results,
    )
