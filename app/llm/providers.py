"""Concrete LLM providers — Gemini, Groq, Ollama. All free."""
from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.llm.base import LLMProvider
from app.logging_conf import get_logger

log = get_logger(__name__)

_RETRY = dict(
    stop=stop_after_attempt(settings.llm_max_retries),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)


class GeminiProvider(LLMProvider):
    """Google Gemini — primary. Free tier at aistudio.google.com/apikey"""

    name = "gemini"

    def __init__(self) -> None:
        import google.generativeai as genai

        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey"
            )
        genai.configure(api_key=settings.gemini_api_key)
        self._genai = genai

    @retry(**_RETRY)
    def complete(self, prompt: str, *, system: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 1024) -> str:
        model = self._genai.GenerativeModel(
            settings.gemini_model,
            system_instruction=system or None,
        )
        resp = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
            request_options={"timeout": settings.llm_timeout_seconds},
        )
        return (resp.text or "").strip()


class GroqProvider(LLMProvider):
    """Groq — fallback if Gemini rate-limits. Free tier, very fast."""

    name = "groq"

    def __init__(self) -> None:
        from groq import Groq

        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set.")
        self._client = Groq(api_key=settings.groq_api_key)

    @retry(**_RETRY)
    def complete(self, prompt: str, *, system: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 1024) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = self._client.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=settings.llm_timeout_seconds,
        )
        return (resp.choices[0].message.content or "").strip()


class OllamaProvider(LLMProvider):
    """Ollama — fully local. Demo insurance: works with zero internet."""

    name = "ollama"

    @retry(**_RETRY)
    def complete(self, prompt: str, *, system: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 1024) -> str:
        payload = {
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system:
            payload["system"] = system

        with httpx.Client(timeout=settings.llm_timeout_seconds) as client:
            r = client.post(f"{settings.ollama_base_url}/api/generate", json=payload)
            r.raise_for_status()
            return (r.json().get("response") or "").strip()


_PROVIDERS = {
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "ollama": OllamaProvider,
}

_instance: LLMProvider | None = None


def get_llm(provider: str | None = None) -> LLMProvider:
    """Return a cached LLM instance for the configured provider."""
    global _instance
    key = (provider or settings.llm_provider).lower()

    if _instance is not None and provider is None:
        return _instance

    if key not in _PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{key}'. Choose from: {list(_PROVIDERS)}"
        )

    llm = _PROVIDERS[key]()
    log.info("llm.initialised", provider=key)
    if provider is None:
        _instance = llm
    return llm
