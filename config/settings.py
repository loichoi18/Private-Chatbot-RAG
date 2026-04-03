"""
Application configuration using Pydantic Settings.
All secrets loaded from environment variables or .env file.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from enum import Enum


class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"
    OLLAMA = "ollama"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class VectorStoreProvider(str, Enum):
    CHROMA = "chroma"
    QDRANT = "qdrant"


class Settings(BaseSettings):
    """Central configuration -- reads from .env automatically."""

    # -- LLM --
    llm_provider: LLMProvider = LLMProvider.OPENAI
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048

    # -- Embedding --
    embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    cohere_api_key: Optional[str] = None

    # -- Vector Store --
    vectorstore_provider: VectorStoreProvider = VectorStoreProvider.CHROMA
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "documents"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "documents"

    # -- Chunking --
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list[str] = Field(default=["\n\n", "\n", ". ", " ", ""])

    # -- Retrieval --
    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.3
    rerank_enabled: bool = False
    rerank_top_k: int = 3

    # -- Ingestion --
    supported_extensions: list[str] = Field(
        default=[".pdf", ".txt", ".md", ".csv", ".html", ".docx"]
    )
    max_file_size_mb: int = 50
    web_scrape_timeout: int = 30

    # -- API --
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_title: str = "Private RAG Chatbot API"
    cors_origins: list[str] = Field(default=["*"])

    # -- Logging --
    log_level: str = "INFO"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


settings = Settings()
