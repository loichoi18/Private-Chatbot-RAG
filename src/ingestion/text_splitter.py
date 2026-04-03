"""
Text Splitter
Splits documents into chunks optimized for embedding and retrieval.
Supports multiple strategies: recursive, semantic, and sentence-based.
"""

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings

logger = logging.getLogger(__name__)


class TextSplitter:
    """
    Chunk documents for vector storage.

    Why chunking matters:
    ─────────────────────
    • Embedding models have token limits (typically 512 tokens).
    • Smaller, focused chunks improve retrieval precision.
    • Overlap ensures context isn't lost at boundaries.
    • Metadata propagation keeps source traceability.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[list[str]] = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.separators = separators or settings.separators

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """
        Split a list of documents into smaller chunks.
        Each chunk inherits the parent document's metadata plus
        chunk-specific fields.
        """
        chunks = self.splitter.split_documents(documents)

        # Enrich metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        logger.info(
            f"Split {len(documents)} documents into {len(chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks

    def split_text(self, text: str) -> list[str]:
        """Split a plain text string into chunks (no metadata)."""
        return self.splitter.split_text(text)
