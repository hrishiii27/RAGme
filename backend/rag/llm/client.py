"""Phase 6 — LLM Clients (Base Interface & Implementations).

Supports:
- Gemini (Google GenAI SDK)
- OpenRouter (OpenAI-compatible HTTP with real-time SSE streaming & free model fallback)
- Ollama (Local open-source models like llama3.2:3b)
"""

from __future__ import annotations

import json
import logging
import os
from typing import AsyncIterator, Iterator, Protocol

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Protocol for uniform LLM invocation and streaming."""

    @property
    def model_name(self) -> str:
        ...

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        ...

    def stream(self, prompt: str, system_prompt: str = "") -> Iterator[str]:
        ...


class GeminiClient(LLMClient):
    """Client for Google Gemini models with primary pro reasoning and automatic flash fallback."""

    def __init__(
        self,
        model_name: str = "gemini-3.1-pro-preview",
        fallback_model: str = "gemini-2.5-flash",
        api_key: str | None = None,
    ):
        self._model_name = model_name
        self.fallback_model = fallback_model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required for GeminiClient")

        from google import genai
        self.client = genai.Client(api_key=self.api_key)

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        from google.genai import types
        config = types.GenerateContentConfig(
            temperature=0.2,
            system_instruction=system_prompt if system_prompt else None,
        )
        try:
            resp = self.client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=config,
            )
            return resp.text or ""
        except Exception as e:
            logger.warning(f"[GeminiClient] {self._model_name} failed ({e}). Falling back to {self.fallback_model}...")
            resp = self.client.models.generate_content(
                model=self.fallback_model,
                contents=prompt,
                config=config,
            )
            return resp.text or ""

    def stream(self, prompt: str, system_prompt: str = "") -> Iterator[str]:
        from google.genai import types
        config = types.GenerateContentConfig(
            temperature=0.2,
            system_instruction=system_prompt if system_prompt else None,
        )
        try:
            response_stream = self.client.models.generate_content_stream(
                model=self._model_name,
                contents=prompt,
                config=config,
            )
            yielded_any = False
            for chunk in response_stream:
                if chunk.text:
                    yielded_any = True
                    yield chunk.text
            if not yielded_any:
                raise RuntimeError(f"No tokens returned from {self._model_name}")
        except Exception as e:
            logger.warning(f"[GeminiClient] {self._model_name} streaming failed ({e}). Falling back to {self.fallback_model}...")
            fallback_stream = self.client.models.generate_content_stream(
                model=self.fallback_model,
                contents=prompt,
                config=config,
            )
            for chunk in fallback_stream:
                if chunk.text:
                    yield chunk.text


class OpenRouterClient(LLMClient):
    """Client for OpenRouter free models with real-time SSE streaming & automatic fallback."""

    def __init__(
        self,
        model_name: str = "nvidia/nemotron-3-super-120b-a12b:free",
        fallback_model: str = "openrouter/free",
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        self._model_name = model_name
        self.fallback_model = fallback_model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouterClient")
        self.base_url = base_url.rstrip("/")
        import httpx
        self.client = httpx.Client(timeout=45.0)

    @property
    def model_name(self) -> str:
        return self._model_name

    def _build_payload(self, model: str, prompt: str, system_prompt: str, stream: bool = False) -> tuple[dict, dict]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/hrishikeshvirupakshi/RAGme",
            "X-Title": "Bharat Guide Official Public Intelligence",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "stream": stream,
        }
        return headers, payload

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        headers, payload = self._build_payload(self._model_name, prompt, system_prompt, stream=False)
        try:
            resp = self.client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            logger.warning(f"[OpenRouter] {self._model_name} returned status {resp.status_code}. Failing over to {self.fallback_model}...")
        except Exception as e:
            logger.warning(f"[OpenRouter] {self._model_name} error: {e}. Failing over to {self.fallback_model}...")

        # Fallback to secondary free model
        fb_headers, fb_payload = self._build_payload(self.fallback_model, prompt, system_prompt, stream=False)
        resp = self.client.post(f"{self.base_url}/chat/completions", headers=fb_headers, json=fb_payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def stream(self, prompt: str, system_prompt: str = "") -> Iterator[str]:
        headers, payload = self._build_payload(self._model_name, prompt, system_prompt, stream=True)
        try:
            yielded = False
            with self.client.stream("POST", f"{self.base_url}/chat/completions", headers=headers, json=payload) as resp:
                if resp.status_code == 200:
                    for line in resp.iter_lines():
                        if line.startswith("data: ") and line != "data: [DONE]":
                            raw_json = line[6:]
                            try:
                                chunk = json.loads(raw_json)
                                delta = chunk["choices"][0]["delta"].get("content", "")
                                if delta:
                                    yielded = True
                                    yield delta
                            except Exception:
                                continue
                    if yielded:
                        return
            logger.warning(f"[OpenRouter] {self._model_name} streaming failed or empty. Falling over to {self.fallback_model}...")
        except Exception as e:
            logger.warning(f"[OpenRouter] {self._model_name} stream error: {e}. Falling over to {self.fallback_model}...")

        # Fallback stream
        fb_headers, fb_payload = self._build_payload(self.fallback_model, prompt, system_prompt, stream=True)
        with self.client.stream("POST", f"{self.base_url}/chat/completions", headers=fb_headers, json=fb_payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    raw_json = line[6:]
                    try:
                        chunk = json.loads(raw_json)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except Exception:
                        continue


class OllamaClient(LLMClient):
    """Client for local Ollama server (e.g. llama3.2:3b)."""

    def __init__(self, model_name: str = "llama3.2:3b", base_url: str = "http://localhost:11434"):
        self._model_name = model_name
        self.base_url = base_url.rstrip("/")
        import httpx
        self.client = httpx.Client(timeout=90.0)

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        resp = self.client.post(f"{self.base_url}/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json().get("response", "")

    def stream(self, prompt: str, system_prompt: str = "") -> Iterator[str]:
        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "system": system_prompt,
            "stream": True,
            "options": {"temperature": 0.2},
        }
        with self.client.stream("POST", f"{self.base_url}/api/generate", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    text = data.get("response", "")
                    if text:
                        yield text


def create_llm_client(provider: str | None = None, model_name: str | None = None) -> LLMClient:
    """Factory creating the configured LLM client."""
    selected_provider = (provider or os.getenv("LLM_PROVIDER", "openrouter" if os.getenv("OPENROUTER_API_KEY") else "gemini")).lower()

    if selected_provider == "openrouter":
        chosen_model = model_name or os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
        return OpenRouterClient(
            model_name=chosen_model,
            fallback_model="openrouter/free",
        )
    elif selected_provider == "ollama":
        return OllamaClient(model_name=model_name or "llama3.2:3b")
    else:
        return GeminiClient(
            model_name=model_name or "gemini-3.1-pro-preview",
            fallback_model="gemini-2.5-flash",
        )
