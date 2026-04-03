"""
Retriever
Advanced retrieval strategies including:
  - Basic similarity search
  - MMR (Maximum Marginal Relevance) for diversity
  - Contextual compression with reranking
  - Hybrid retrieval (combine multiple strategies)
"""

import logging
from typing import Optional

from langchain_core.documents import Document

from config.settings import settings
from src.vectorstore.store import VectorStoreManager

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retriever wraps the vector store with advanced search strategies.

    Retrieval quality is the #1 factor in RAG performance.
    A good retriever finds the most relevant chunks while
    minimizing noise that confuses the LLM.
    """

    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        strategy: str = "similarity",
        top_k: Optional[int] = None,
    ) -> list[Document]:
        """
        Retrieve relevant documents using the specified strategy.

        Strategies:
          similarity — cosine similarity (fast, default)
          mmr        — maximum marginal relevance (diverse)
          rerank     — retrieve more, then rerank with cross-encoder
        """
        top_k = top_k or settings.retrieval_top_k

        if strategy == "similarity":
            return self._similarity_retrieve(query, top_k)
        elif strategy == "mmr":
            return self._mmr_retrieve(query, top_k)
        elif strategy == "rerank":
            return self._rerank_retrieve(query, top_k)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _similarity_retrieve(self, query: str, top_k: int) -> list[Document]:
        """Standard cosine similarity search."""
        results = self.vector_store.similarity_search_with_scores(query, k=top_k)
        documents = []
        for doc, score in results:
            doc.metadata["relevance_score"] = round(score, 4)
            documents.append(doc)
        logger.info(f"Similarity search returned {len(documents)} results")
        return documents

    def _mmr_retrieve(self, query: str, top_k: int) -> list[Document]:
        """
        MMR balances relevance and diversity.
        Prevents returning 5 chunks that all say the same thing.
        """
        documents = self.vector_store.max_marginal_relevance_search(
            query, k=top_k, fetch_k=top_k * 4, lambda_mult=0.5
        )
        logger.info(f"MMR search returned {len(documents)} results")
        return documents

    def _rerank_retrieve(self, query: str, top_k: int) -> list[Document]:
        """
        Two-stage retrieval:
          Stage 1: Retrieve more candidates with fast similarity search
          Stage 2: Rerank with a cross-encoder for better precision

        Cross-encoders are slower but much more accurate than
        bi-encoder (embedding) similarity because they see
        query and document together.
        """
        # Stage 1: Over-fetch candidates
        candidates = self.vector_store.similarity_search(
            query, k=top_k * 4
        )
        logger.info(f"Rerank stage 1: fetched {len(candidates)} candidates")

        if not candidates:
            return []

        # Stage 2: Rerank
        try:
            from sentence_transformers import CrossEncoder

            reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pairs = [(query, doc.page_content) for doc in candidates]
            scores = reranker.predict(pairs)

            # Sort by rerank score (descending)
            scored = list(zip(candidates, scores))
            scored.sort(key=lambda x: x[1], reverse=True)

            reranked = []
            for doc, score in scored[: settings.rerank_top_k]:
                doc.metadata["rerank_score"] = round(float(score), 4)
                reranked.append(doc)

            logger.info(f"Rerank stage 2: selected top {len(reranked)} documents")
            return reranked

        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Falling back to similarity."
            )
            return candidates[:top_k]

    def retrieve_with_context(
        self, query: str, strategy: str = "similarity", top_k: Optional[int] = None
    ) -> str:
        """
        Retrieve and format documents into a context string
        ready for the LLM prompt.
        """
        documents = self.retrieve(query, strategy=strategy, top_k=top_k)
        if not documents:
            return "No relevant documents found."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            page_str = f" (page {page})" if page else ""
            context_parts.append(
                f"[Document {i} — {source}{page_str}]\n{doc.page_content}"
            )

        return "\n\n---\n\n".join(context_parts)
