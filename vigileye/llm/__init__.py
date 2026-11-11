"""OpenAI-compatible LLM access with role routing, fallbacks, caching and usage accounting."""

from .router import ChatResult, LLMError, LLMRouter, Provider, RouteEntry, extract_json

__all__ = ["LLMRouter", "ChatResult", "LLMError", "Provider", "RouteEntry", "extract_json"]
