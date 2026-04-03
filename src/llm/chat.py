"""
LLM Factory
Creates LLM instances for OpenAI, Anthropic, or local Ollama models.
"""

import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel

from config.settings import settings, LLMProvider

logger = logging.getLogger(__name__)


class LLMFactory:
    """
    Factory that returns the configured LLM.

    The LLM is the "brain" that synthesizes retrieved context
    into natural-language answers.
    """

    @staticmethod
    def create(
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> BaseChatModel:
        provider = provider or settings.llm_provider
        model = model or settings.llm_model
        temperature = temperature if temperature is not None else settings.llm_temperature

        logger.info(f"Creating LLM: provider={provider}, model={model}, temp={temperature}")

        if provider == LLMProvider.OPENAI:
            return LLMFactory._create_openai(model, temperature)
        elif provider == LLMProvider.ANTHROPIC:
            return LLMFactory._create_anthropic(model, temperature)
        elif provider == LLMProvider.OLLAMA:
            return LLMFactory._create_ollama(model, temperature)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    @staticmethod
    def _create_openai(model: str, temperature: float) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=settings.llm_max_tokens,
            openai_api_key=settings.openai_api_key,
        )

    @staticmethod
    def _create_anthropic(model: str, temperature: float) -> BaseChatModel:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=settings.llm_max_tokens,
            anthropic_api_key=settings.anthropic_api_key,
        )

    @staticmethod
    def _create_ollama(model: str, temperature: float) -> BaseChatModel:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=settings.ollama_base_url,
        )
