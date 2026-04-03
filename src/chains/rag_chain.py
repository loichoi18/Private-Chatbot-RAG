"""
RAG Chain
Orchestrates the full Retrieval-Augmented Generation pipeline:
  Query → Retrieve → Build Prompt → LLM → Answer

Supports conversation history for multi-turn chat.
"""

import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


# ── System Prompt ────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based strictly on the provided context documents.

RULES:
1. Answer ONLY based on the provided context. Do not use prior knowledge.
2. If the context does not contain enough information, say: "I don't have enough information in the provided documents to answer this question."
3. Cite the source document when possible (e.g., "According to [Document 1]...").
4. Be concise but thorough. Use bullet points for complex answers.
5. If asked about something outside the documents, politely redirect.
6. Maintain a professional, helpful tone.

CONTEXT:
{context}
"""

CONDENSE_QUESTION_PROMPT = """Given the following conversation history and a follow-up question, rephrase the follow-up question as a standalone question that captures the full intent.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""


class RAGChain:
    """
    The core RAG pipeline.

    Flow:
    ─────
    1. User asks a question
    2. (Optional) Condense with chat history into standalone query
    3. Retrieve relevant document chunks from vector store
    4. Build a prompt with context + question
    5. LLM generates a grounded answer
    6. Return answer with source references
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: BaseChatModel,
        retrieval_strategy: str = "similarity",
    ):
        self.retriever = retriever
        self.llm = llm
        self.retrieval_strategy = retrieval_strategy

    def query(
        self,
        question: str,
        chat_history: Optional[list[dict]] = None,
        retrieval_strategy: Optional[str] = None,
    ) -> dict:
        """
        Run the full RAG pipeline.

        Args:
            question:           The user's question.
            chat_history:       List of {"role": "user"|"assistant", "content": "..."}.
            retrieval_strategy: Override the default retrieval strategy.

        Returns:
            {
                "answer": str,
                "sources": list[dict],
                "context_used": str,
            }
        """
        strategy = retrieval_strategy or self.retrieval_strategy

        # Step 1: Condense question if there's chat history
        standalone_question = question
        if chat_history:
            standalone_question = self._condense_question(question, chat_history)
            logger.info(f"Condensed question: {standalone_question}")

        # Step 2: Retrieve relevant documents
        context = self.retriever.retrieve_with_context(
            standalone_question, strategy=strategy
        )
        source_docs = self.retriever.retrieve(standalone_question, strategy=strategy)

        # Step 3: Build messages
        messages = [
            SystemMessage(content=RAG_SYSTEM_PROMPT.format(context=context)),
        ]

        # Add chat history
        if chat_history:
            for msg in chat_history[-10:]:  # Last 10 turns
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=question))

        # Step 4: Generate answer
        logger.info(f"Sending {len(messages)} messages to LLM")
        response = self.llm.invoke(messages)
        answer = response.content

        # Step 5: Extract source metadata
        sources = []
        seen = set()
        for doc in source_docs:
            source_id = doc.metadata.get("source", "unknown")
            if source_id not in seen:
                seen.add(source_id)
                sources.append(
                    {
                        "source": source_id,
                        "page": doc.metadata.get("page"),
                        "filename": doc.metadata.get("filename"),
                        "relevance_score": doc.metadata.get("relevance_score"),
                    }
                )

        return {
            "answer": answer,
            "sources": sources,
            "context_used": context,
            "standalone_question": standalone_question,
        }

    def _condense_question(self, question: str, chat_history: list[dict]) -> str:
        """Rephrase a follow-up question as a standalone question."""
        history_str = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in chat_history[-6:]
        )
        prompt = CONDENSE_QUESTION_PROMPT.format(
            chat_history=history_str, question=question
        )
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()

    async def aquery(
        self,
        question: str,
        chat_history: Optional[list[dict]] = None,
        retrieval_strategy: Optional[str] = None,
    ) -> dict:
        """Async version of query()."""
        strategy = retrieval_strategy or self.retrieval_strategy

        standalone_question = question
        if chat_history:
            standalone_question = self._condense_question(question, chat_history)

        context = self.retriever.retrieve_with_context(
            standalone_question, strategy=strategy
        )
        source_docs = self.retriever.retrieve(standalone_question, strategy=strategy)

        messages = [
            SystemMessage(content=RAG_SYSTEM_PROMPT.format(context=context)),
        ]
        if chat_history:
            for msg in chat_history[-10:]:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=question))

        response = await self.llm.ainvoke(messages)

        sources = []
        seen = set()
        for doc in source_docs:
            source_id = doc.metadata.get("source", "unknown")
            if source_id not in seen:
                seen.add(source_id)
                sources.append(
                    {
                        "source": source_id,
                        "page": doc.metadata.get("page"),
                        "filename": doc.metadata.get("filename"),
                    }
                )

        return {
            "answer": response.content,
            "sources": sources,
            "context_used": context,
        }
