"""Pydantic v2 schemas for the RAG API."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Citizen inquiry or question")
    history: list[ChatMessage] = Field(default_factory=list, description="Previous conversation turns")
    stream: bool = Field(default=True, description="Whether to stream the response via SSE")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve")


class SourceChunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    url: str
    score: float
    snippet: str
    category: str | None = None
    department: str | None = None


class ChatResponse(BaseModel):
    answer: str
    standalone_question: str
    sources: list[SourceChunk]
    follow_up_questions: list[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    strategy: Literal["hybrid_rrf", "bm25", "dense"] = "hybrid_rrf"


class SearchResponse(BaseModel):
    query: str
    count: int
    strategy: str
    results: list[SourceChunk]


class FeedbackRequest(BaseModel):
    message_id: str | None = None
    query: str
    answer: str
    rating: Literal["positive", "negative"]
    critique: str | None = None
    sources_cited: list[str] = Field(default_factory=list)


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: str


class StatsResponse(BaseModel):
    status: str
    total_documents: int
    total_chunks: int
    chroma_collection: str
    vector_store: str | None = None
    llm_model: str | None = None
    dense_embedder: str
    retrieval_strategy: str
    prompt_winner: str
    device: str
