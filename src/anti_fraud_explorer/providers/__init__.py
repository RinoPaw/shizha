"""External provider boundaries used by the assistant core."""

from .llm import LLMProvider, OpenAICompatibleLLM

__all__ = ["LLMProvider", "OpenAICompatibleLLM"]
