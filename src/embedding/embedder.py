"""
Embedding Factory
Creates embedding instances for OpenAI, HuggingFace, Cohere, or Ollama.
Abstracts provider differences behind a common interface.
"""

import logging
from typing import Optional

from langchain_core.embeddings import Embeddings

from config.settings import settings, EmbeddingProvider

logger = logging.getLogger(__name__)


class EmbeddingFactory:
    """
    Factory that returns the right embedding model based on config.

    Embedding models convert text -> dense vectors so that
    semantically similar texts land near each other in vector space.
    """

    @staticmethod
    def create(
        provider: Optional[EmbeddingProvider] = None,
        model: Optional[str] = None,
    ) -> Embeddings:
        provider = provider or settings.embedding_provider
        model = model or settings.embedding_model

        logger.info(f"Creating embedding: provider={provider}, model={model}")

        if provider == EmbeddingProvider.OPENAI:
            return EmbeddingFactory._create_openai(model)
        elif provider == EmbeddingProvider.HUGGINGFACE:
            return EmbeddingFactory._create_huggingface(model)
        elif provider == EmbeddingProvider.COHERE:
            return EmbeddingFactory._create_cohere(model)
        elif provider == EmbeddingProvider.OLLAMA:
            return EmbeddingFactory._create_ollama(model)
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

    @staticmethod
    def _create_openai(model: str) -> Embeddings:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=model,
            openai_api_key=settings.openai_api_key,
        )

    @staticmethod
    def _create_huggingface(model: str) -> Embeddings:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    @staticmethod
    def _create_cohere(model: str) -> Embeddings:
        from langchain_cohere import CohereEmbeddings

        return CohereEmbeddings(
            model=model,
            cohere_api_key=settings.cohere_api_key,
        )

    @staticmethod
    def _create_ollama(model: str) -> Embeddings:
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=model,
            base_url=settings.ollama_base_url,
        )
