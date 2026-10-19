"""LangChain ChatModel Adapter for internal LLMClient implementations.

Allows our existing Gemini, OpenRouter, and Ollama clients to be seamlessly
used inside any LangChain LCEL chain with full streaming support.
"""

from __future__ import annotations

from typing import Any, Iterator, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

from rag.llm.client import LLMClient


class LangChainLLMAdapter(BaseChatModel):
    """Bridges custom LLMClient instances to LangChain's BaseChatModel interface."""

    client: Any = Field(description="Internal LLMClient instance")

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return f"custom-{self.client.model_name}"

    def _extract_prompts(self, messages: List[BaseMessage]) -> tuple[str, str]:
        """Separates system prompt from conversational user/assistant prompt."""
        system_parts = []
        user_parts = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_parts.append(str(msg.content))
            elif isinstance(msg, HumanMessage):
                user_parts.append(str(msg.content))
            elif isinstance(msg, AIMessage):
                user_parts.append(f"Assistant: {msg.content}")
            else:
                user_parts.append(str(msg.content))

        system_prompt = "\n\n".join(system_parts)
        prompt = "\n\n".join(user_parts)
        return prompt, system_prompt

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt, system_prompt = self._extract_prompts(messages)
        output_text = self.client.generate(prompt, system_prompt=system_prompt)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=output_text))]
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        prompt, system_prompt = self._extract_prompts(messages)
        for token in self.client.stream(prompt, system_prompt=system_prompt):
            yield ChatGenerationChunk(message=AIMessageChunk(content=token))
