"""Chat and streaming SSE endpoint."""

from __future__ import annotations

import json
from typing import AsyncIterator

import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas import ChatRequest, ChatResponse, SourceChunk
from app.services import extract_follow_ups, get_pipeline
from rag.eval.tracer import RAGTracer
from rag.generate.prompt import SYSTEM_PROMPT_WINNER, build_rag_prompt, format_context

router = APIRouter()


def format_history(history: list) -> str:
    lines = []
    for msg in history:
        role = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


@router.post("/chat", response_model=None)
async def chat_endpoint(request: ChatRequest):
    start_time = time.time()
    pipeline = get_pipeline()
    chat_history_str = format_history(request.history)

    # 1. Condense Question (if history is present)
    standalone_q = pipeline.condense_question(chat_history_str, request.message)

    # 2. Hybrid Retrieval (RRF)
    fused_chunks = pipeline.retrieve(standalone_q, top_k=request.top_k)

    sources = [
        SourceChunk(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            title=c.metadata.get("title") or "Official Bharat Document",
            url=c.metadata.get("url") or "",
            score=round(float(score), 4),
            snippet=c.text[:280] + ("..." if len(c.text) > 280 else ""),
        )
        for c, score in fused_chunks
    ]

    context_str = format_context(fused_chunks)
    prompt = build_rag_prompt(standalone_q, context_str)

    # If non-streaming requested:
    if not request.stream:
        raw_answer = pipeline.llm.generate(prompt, system_prompt=SYSTEM_PROMPT_WINNER)
        clean_answer, follow_ups = extract_follow_ups(raw_answer)
        return ChatResponse(
            answer=clean_answer,
            standalone_question=standalone_q,
            sources=sources,
            follow_up_questions=follow_ups,
        )

    # Streaming mode via Server-Sent Events (SSE)
    async def sse_event_generator():
        # Event 1: Send initial metadata (standalone question + sources)
        metadata_payload = {
            "standalone_question": standalone_q,
            "sources": [s.model_dump() for s in sources],
        }
        yield f"event: metadata\ndata: {json.dumps(metadata_payload)}\n\n"

        # Stream LLM tokens
        full_text_accum = []
        try:
            token_stream = pipeline.llm.stream(prompt, system_prompt=SYSTEM_PROMPT_WINNER)
            for token in token_stream:
                full_text_accum.append(token)
                yield f"event: token\ndata: {json.dumps({'delta': token})}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
            return

        # Post-process full answer for follow-ups
        full_text = "".join(full_text_accum)
        clean_answer, follow_ups = extract_follow_ups(full_text)
        latency = round(time.time() - start_time, 3)

        # Log trace to Langfuse (if configured)
        trace_id = RAGTracer.trace_query(
            question=request.message,
            retrieved_chunks=[s.model_dump() for s in sources],
            answer=clean_answer,
            latency_seconds=latency,
        )

        # Event 2: Send structured follow-up questions
        yield f"event: follow_ups\ndata: {json.dumps({'follow_up_questions': follow_ups})}\n\n"

        # Event 3: Done signal with complete clean answer & trace_id
        done_payload = {
            "status": "completed",
            "clean_answer": clean_answer,
            "latency": latency,
            "trace_id": trace_id,
        }
        yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
