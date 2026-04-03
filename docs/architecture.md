# Architecture Overview

## System Design

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline — the industry-standard pattern for building AI systems that answer questions using private, domain-specific data instead of relying on the LLM's training data.

## Why RAG?

| Approach | Pros | Cons |
|---|---|---|
| Fine-tuning | Bakes knowledge into model weights | Expensive, slow, needs retraining for updates |
| Prompt stuffing | Simple | Limited by context window, no scalability |
| **RAG** | **Scalable, updatable, cost-effective, traceable** | **Requires retrieval infrastructure** |

RAG is the preferred approach for enterprise use cases because it keeps data private, provides source citations, and can be updated instantly by adding new documents.

## Pipeline Flow

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  Documents   │────▶│  Text        │────▶│  Embedding    │
│  (PDF, Web,  │     │  Splitter    │     │  Model        │
│   DB, etc.)  │     │  (Chunking)  │     │  (Vectorize)  │
└─────────────┘     └──────────────┘     └───────┬───────┘
                                                  │
                                                  ▼
                                         ┌───────────────┐
                                         │  Vector Store  │
                                         │  (ChromaDB)    │
                                         └───────┬───────┘
                                                  │
┌─────────────┐     ┌──────────────┐              │
│   Answer +   │◀───│  LLM         │◀─────────────┘
│   Sources    │     │  (Generate)  │     ┌───────────────┐
└─────────────┘     └──────┬───────┘     │  Retriever     │
                           │             │  (Search +     │
                           └─────────────│   Rerank)      │
                                         └───────┬───────┘
                                                  │
                                         ┌───────────────┐
                                         │  User Query    │
                                         └───────────────┘
```

## Component Details

### 1. Ingestion Layer (`src/ingestion/`)

Responsible for loading raw data from multiple sources and converting it into a unified `Document` format.

- **PDFLoader** — Extracts text from PDFs using PyPDF2, pdfplumber (for tables), and OCR fallback (for scanned docs)
- **WebLoader** — Scrapes web pages, sitemaps, and recursive crawling with BeautifulSoup
- **DatabaseLoader** — Connects to SQL databases (SQLite, PostgreSQL, MySQL) via SQLAlchemy
- **FileLoader** — Handles TXT, Markdown, CSV, HTML, DOCX files
- **TextSplitter** — Chunks documents with configurable size/overlap using recursive character splitting

### 2. Embedding Layer (`src/embedding/`)

Converts text chunks into dense vector representations (embeddings).

Supported providers:
- **OpenAI** (`text-embedding-3-small`) — Best quality-to-cost ratio
- **HuggingFace** (`all-MiniLM-L6-v2`) — Free, runs locally
- **Cohere** (`embed-english-v3.0`) — Strong multilingual support

### 3. Vector Store (`src/vectorstore/`)

Stores embeddings and enables fast approximate nearest-neighbor (ANN) search.

- **ChromaDB** (default) — Lightweight, file-based, zero infrastructure
- **Qdrant** (optional) — Production-grade, distributed vector database

### 4. Retrieval Layer (`src/retrieval/`)

Finds the most relevant document chunks for a given query.

Strategies:
- **Similarity** — Standard cosine similarity (fast, default)
- **MMR** — Maximum Marginal Relevance (balances relevance + diversity)
- **Rerank** — Two-stage: over-fetch with embeddings, then rerank with cross-encoder for higher precision

### 5. LLM Layer (`src/llm/`)

Generates natural-language answers grounded in retrieved context.

Supported providers:
- **OpenAI** (GPT-4o-mini, GPT-4o)
- **Anthropic** (Claude 3.5 Sonnet, Claude 3 Opus)
- **Ollama** (Llama 3.1, Mistral — fully local, free)

### 6. RAG Chain (`src/chains/`)

Orchestrates the full pipeline:
1. Condenses follow-up questions using chat history
2. Retrieves relevant context
3. Constructs a grounded prompt with instructions
4. Invokes the LLM
5. Returns answer + source citations

### 7. API & UI

- **FastAPI** (`src/api/`) — REST endpoints for chat, ingestion, and management
- **Streamlit** (`src/ui/`) — Interactive chat interface with sidebar document management

## Key Design Decisions

1. **Provider abstraction** — Factory patterns let you swap OpenAI for Ollama (or any provider) with a single env var change
2. **Metadata propagation** — Source, page number, and relevance scores travel with every chunk for full traceability
3. **Configurable chunking** — Chunk size/overlap are tunable because optimal values depend on your document types
4. **Multi-strategy retrieval** — Different use cases need different strategies (customer support → similarity; research → rerank)
5. **Conversation memory** — Chat history enables follow-up questions without losing context
