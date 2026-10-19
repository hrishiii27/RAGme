"""LangChain Expression Language (LCEL) implementation of RAG Pipeline.

Demonstrates the declarative composition of RAG components using LangChain's
Runnable protocol (| pipe syntax), comparing with our custom from-scratch pipeline.
"""

from __future__ import annotations

from operator import itemgetter
from typing import Any, Iterator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from rag.generate.prompt import (
    CONDENSE_PROMPT,
    SYSTEM_PROMPT_FOLLOWUP,
    format_context,
)
from rag.lcel.adapter import LangChainLLMAdapter
from rag.pipeline import RAGPipeline
from rag.types import Chunk


class LCELRAGPipeline:
    """LangChain LCEL implementation of the Bharat-RAG pipeline."""

    def __init__(self, base_pipeline: RAGPipeline):
        self.base_pipeline = base_pipeline
        self.llm_adapter = LangChainLLMAdapter(client=base_pipeline.llm)
        self._build_chains()

    def _build_chains(self) -> None:
        """Constructs the LCEL runnables using the declarative pipe (|) operator."""
        
        # 1. Question Condensation Runnable
        condense_prompt = PromptTemplate.from_template(CONDENSE_PROMPT)
        self.condense_chain = (
            condense_prompt
            | self.llm_adapter
            | StrOutputParser()
        )

        def _resolve_standalone(inputs: dict[str, Any]) -> str:
            history = (inputs.get("chat_history") or "").strip()
            q = inputs.get("question", "")
            if not history:
                return q
            condensed = self.condense_chain.invoke({"chat_history": history, "question": q})
            return condensed.strip() or q

        # 2. Retrieval Runnable
        def _retrieve_chunks(inputs: dict[str, Any]) -> list[tuple[Chunk, float]]:
            query = inputs["standalone_question"]
            top_k = inputs.get("top_k", 5)
            return self.base_pipeline.retrieve(query, top_k=top_k)

        # 3. Context Formatter Runnable
        def _format_docs(inputs: dict[str, Any]) -> str:
            chunks = inputs["retrieved_chunks"]
            return format_context(chunks)

        # 4. Prompt Template
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("human", "Answer the question based only on the following context:\n\n{context}\n\nQuestion: {standalone_question}\nAnswer:"),
        ])

        # 5. Full End-to-End LCEL Runnable Pipeline
        self.rag_chain = (
            RunnablePassthrough.assign(
                standalone_question=RunnableLambda(_resolve_standalone),
                system_prompt=lambda x: x.get("system_prompt", SYSTEM_PROMPT_FOLLOWUP),
            )
            | RunnablePassthrough.assign(
                retrieved_chunks=RunnableLambda(_retrieve_chunks)
            )
            | RunnablePassthrough.assign(
                context=RunnableLambda(_format_docs)
            )
            | RunnablePassthrough.assign(
                answer=(
                    rag_prompt
                    | self.llm_adapter
                    | StrOutputParser()
                )
            )
        )

        # Streamable generation sub-chain
        self.stream_chain = (
            rag_prompt
            | self.llm_adapter
            | StrOutputParser()
        )

    def answer(
        self,
        question: str,
        chat_history: str = "",
        system_prompt: str = SYSTEM_PROMPT_FOLLOWUP,
    ) -> dict[str, Any]:
        """Invokes the LCEL pipeline synchronously."""
        result = self.rag_chain.invoke({
            "question": question,
            "chat_history": chat_history,
            "system_prompt": system_prompt,
        })
        return {
            "question": question,
            "standalone_question": result["standalone_question"],
            "answer": result["answer"],
            "retrieved_chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "text": c.text,
                    "score": s,
                }
                for c, s in result["retrieved_chunks"]
            ],
        }

    def answer_stream(
        self,
        question: str,
        chat_history: str = "",
        system_prompt: str = SYSTEM_PROMPT_FOLLOWUP,
    ) -> Iterator[str]:
        """Streams tokens from the LCEL generation chain."""
        history = chat_history.strip()
        if history:
            condensed = self.condense_chain.invoke({"chat_history": history, "question": question}).strip()
            standalone_q = condensed or question
        else:
            standalone_q = question

        chunks = self.base_pipeline.retrieve(standalone_q, top_k=5)
        ctx = format_context(chunks)

        for token in self.stream_chain.stream({
            "system_prompt": system_prompt,
            "context": ctx,
            "standalone_question": standalone_q,
        }):
            yield token
