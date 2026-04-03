"""
Vector Store Manager
Wraps ChromaDB (and optionally Qdrant) for storing and querying embeddings.
Handles collection management, upserts, and similarity search.
"""

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config.settings import settings, VectorStoreProvider

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages the vector database lifecycle.

    Vector databases store document embeddings and enable fast
    approximate nearest-neighbor (ANN) search for retrieval.
    """

    def __init__(
        self,
        embedding: Embeddings,
        provider: Optional[VectorStoreProvider] = None,
    ):
        self.embedding = embedding
        self.provider = provider or settings.vectorstore_provider
        self._store = None

    @property
    def store(self):
        if self._store is None:
            self._store = self._initialize_store()
        return self._store

    def _initialize_store(self):
        if self.provider == VectorStoreProvider.CHROMA:
            return self._init_chroma()
        elif self.provider == VectorStoreProvider.QDRANT:
            return self._init_qdrant()
        else:
            raise ValueError(f"Unknown vector store provider: {self.provider}")

    def _init_chroma(self):
        from langchain_chroma import Chroma

        logger.info(
            f"Initializing ChromaDB at {settings.chroma_persist_dir} "
            f"(collection: {settings.chroma_collection_name})"
        )
        return Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=self.embedding,
            persist_directory=settings.chroma_persist_dir,
        )

    def _init_qdrant(self):
        from langchain_qdrant import Qdrant
        from qdrant_client import QdrantClient

        logger.info(f"Connecting to Qdrant at {settings.qdrant_url}")
        client = QdrantClient(url=settings.qdrant_url)
        return Qdrant(
            client=client,
            collection_name=settings.qdrant_collection_name,
            embeddings=self.embedding,
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        """
        Add documents to the vector store.
        Returns the list of document IDs.
        """
        if not documents:
            logger.warning("No documents to add")
            return []

        logger.info(f"Adding {len(documents)} documents to vector store")
        ids = self.store.add_documents(documents)
        logger.info(f"Successfully added {len(ids)} documents")
        return ids

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> list[Document]:
        """Basic similarity search."""
        k = k or settings.retrieval_top_k
        return self.store.similarity_search(query, k=k)

    def similarity_search_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> list[tuple[Document, float]]:
        """Similarity search that also returns relevance scores."""
        k = k or settings.retrieval_top_k
        return self.store.similarity_search_with_score(query, k=k)

    def max_marginal_relevance_search(
        self,
        query: str,
        k: Optional[int] = None,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> list[Document]:
        """
        MMR search balances relevance with diversity.
        Useful when you want varied context, not just
        the top-k most similar (potentially redundant) chunks.
        """
        k = k or settings.retrieval_top_k
        return self.store.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
        )

    def delete_collection(self):
        """Delete the entire collection. Use with caution."""
        if self.provider == VectorStoreProvider.CHROMA:
            self.store._client.delete_collection(settings.chroma_collection_name)
            self._store = None
            logger.info("Collection deleted")

    def get_collection_stats(self) -> dict:
        """Return basic stats about the collection."""
        if self.provider == VectorStoreProvider.CHROMA:
            collection = self.store._collection
            return {
                "name": collection.name,
                "count": collection.count(),
            }
        return {"provider": self.provider, "status": "connected"}

    def as_retriever(self, **kwargs):
        """Return a LangChain-compatible retriever."""
        search_kwargs = {
            "k": settings.retrieval_top_k,
        }
        search_kwargs.update(kwargs.get("search_kwargs", {}))
        return self.store.as_retriever(
            search_type=kwargs.get("search_type", "similarity"),
            search_kwargs=search_kwargs,
        )
