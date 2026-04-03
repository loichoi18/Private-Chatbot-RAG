"""
FastAPI REST API
Exposes the RAG chatbot as HTTP endpoints for integration
with any frontend, mobile app, or third-party service.
"""

import logging
import os
import shutil
import tempfile
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.settings import settings
from src.embedding.embedder import EmbeddingFactory
from src.vectorstore.store import VectorStoreManager
from src.retrieval.retriever import Retriever
from src.llm.chat import LLMFactory
from src.chains.rag_chain import RAGChain
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.file_loader import FileLoader
from src.ingestion.web_loader import WebLoader
from src.ingestion.text_splitter import TextSplitter

logger = logging.getLogger(__name__)

# ── App Setup ────────────────────────────────────────────────
app = FastAPI(
    title=settings.api_title,
    description="Private RAG Chatbot — ask questions about your own documents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Globals (initialized on startup) ────────────────────────
embedding = None
vector_store = None
retriever = None
llm = None
rag_chain = None
splitter = TextSplitter()


@app.on_event("startup")
async def startup():
    global embedding, vector_store, retriever, llm, rag_chain
    embedding = EmbeddingFactory.create()
    vector_store = VectorStoreManager(embedding)
    retriever = Retriever(vector_store)
    llm = LLMFactory.create()
    rag_chain = RAGChain(retriever, llm)
    logger.info("RAG pipeline initialized")


# ── Request / Response Models ────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[list[dict]] = None
    retrieval_strategy: Optional[str] = "similarity"


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


class IngestURLRequest(BaseModel):
    urls: list[str]


class IngestTextRequest(BaseModel):
    text: str
    metadata: Optional[dict] = None


# ── Endpoints ────────────────────────────────────────────────
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a question and get an answer grounded in your documents."""
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    result = await rag_chain.aquery(
        question=request.question,
        chat_history=request.chat_history,
        retrieval_strategy=request.retrieval_strategy,
    )

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
    )


@app.post("/api/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    """Upload a file (PDF, TXT, MD, CSV, DOCX, HTML) to the knowledge base."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in settings.supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {settings.supported_extensions}",
        )

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        if ext == ".pdf":
            loader = PDFLoader()
            documents = loader.load(tmp_path)
        else:
            loader = FileLoader()
            documents = loader.load(tmp_path)

        # Override source metadata with original filename
        for doc in documents:
            doc.metadata["source"] = file.filename
            doc.metadata["filename"] = file.filename

        chunks = splitter.split(documents)
        ids = vector_store.add_documents(chunks)

        return {
            "status": "success",
            "filename": file.filename,
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "ids_stored": len(ids),
        }
    finally:
        os.unlink(tmp_path)


@app.post("/api/ingest/url")
async def ingest_url(request: IngestURLRequest):
    """Scrape one or more URLs and add to the knowledge base."""
    loader = WebLoader()
    documents = loader.load_urls(request.urls)

    if not documents:
        raise HTTPException(status_code=400, detail="No content extracted from URLs")

    chunks = splitter.split(documents)
    ids = vector_store.add_documents(chunks)

    return {
        "status": "success",
        "urls_processed": len(request.urls),
        "documents_loaded": len(documents),
        "chunks_created": len(chunks),
    }


@app.post("/api/ingest/text")
async def ingest_text(request: IngestTextRequest):
    """Directly ingest raw text into the knowledge base."""
    from langchain_core.documents import Document

    doc = Document(
        page_content=request.text,
        metadata=request.metadata or {"source": "direct_input", "loader": "text"},
    )
    chunks = splitter.split([doc])
    ids = vector_store.add_documents(chunks)

    return {
        "status": "success",
        "chunks_created": len(chunks),
    }


@app.get("/api/stats")
async def get_stats():
    """Get vector store statistics."""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    return vector_store.get_collection_stats()


@app.delete("/api/collection")
async def delete_collection():
    """Delete all documents from the knowledge base."""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    vector_store.delete_collection()
    return {"status": "collection deleted"}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}
