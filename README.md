<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-0.3-green?style=for-the-badge&logo=chainlink&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector_Store-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Ollama-Local_LLM-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
</p>

# 🔒 Private RAG Chatbot

**Build an AI chatbot that answers questions using YOUR private documents — not the internet.**

A production-ready Retrieval-Augmented Generation (RAG) system that ingests PDFs, web pages, databases, and text files into a vector store, then uses an LLM to generate accurate, source-cited answers grounded entirely in your data.
> 🚀 **[Live Demo on Hugging Face](https://huggingface.co/spaces/Loinhoi/private-rag-chatbot)** — Try it now, no setup required!
> **Use Cases:** Customer Support Automation · Internal Knowledge Base · Employee Onboarding · Compliance Q&A · Technical Documentation Assistant

---

## 🧠 What This Project Demonstrates

This is not a wrapper around ChatGPT. This project demonstrates deep understanding of the AI/ML stack required to solve real business problems:

| Skill | Implementation |
|---|---|
| **Document Processing** | Multi-format ingestion (PDF, web scraping, SQL databases, CSV, DOCX, HTML) with fallback strategies including OCR |
| **Text Chunking** | Recursive character splitting with configurable size/overlap to optimize retrieval quality |
| **Embedding Models** | Abstracted factory supporting OpenAI, HuggingFace (free/local), Cohere, and Ollama embeddings |
| **Vector Databases** | ChromaDB (lightweight) and Qdrant (production-grade) with collection management |
| **Retrieval Strategies** | Similarity search, MMR (diversity), and two-stage reranking with cross-encoders |
| **LLM Integration** | Multi-provider support (OpenAI, Anthropic, Ollama/local) with grounded prompting |
| **Conversation Memory** | Multi-turn chat with question condensation for follow-up queries |
| **API Design** | RESTful FastAPI with async support, file upload, and OpenAPI documentation |
| **Deployment** | Docker Compose setup with separate API and UI services |

---

## 📐 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                          │
│   📄 PDF   🌐 Web Pages   🗄️ Database   📝 Text/CSV/DOCX   │
└──────────┬───────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐    ┌──────────────────────┐
│   Document Loaders   │───▶│   Text Splitter      │
│   (Parse & Extract)  │    │   (Chunk Documents)  │
└──────────────────────┘    └──────────┬───────────┘
                                       │
                                       ▼
                            ┌──────────────────────┐
                            │   Embedding Model    │
                            │   (Text → Vectors)   │
                            └──────────┬───────────┘
                                       │
                                       ▼
                            ┌──────────────────────┐
                            │   Vector Store       │
                            │   (ChromaDB/Qdrant)  │
                            └──────────┬───────────┘
                                       │
     User Question ───▶ Retriever ─────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   RAG Chain   │
                    │  Context +    │──────▶  💬 Answer
                    │  Question +   │        + 📎 Sources
                    │  LLM          │
                    └───────────────┘
```

> Full architecture documentation: [`docs/architecture.md`](docs/architecture.md)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- An API key from **one** of: OpenAI, Anthropic, or a local [Ollama](https://ollama.ai) installation (free)

### 1. Clone & Install

```bash
git clone https://github.com/loichoi18/Private-Chatbot-RAG.git
cd Private-Chatbot-RAG

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure

**Option A — Fully free with Ollama (recommended to try first):**

```bash
# Install Ollama from https://ollama.ai then:
ollama pull llama3.2
ollama pull nomic-embed-text

cp .env.example .env
```

Edit `.env` and set:
```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
OLLAMA_BASE_URL=http://localhost:11434
```

**Option B — With OpenAI (best quality):**

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-your-key-here
```

### 3. Ingest Documents

```bash
# Ingest sample documents
python scripts/ingest.py --source ./data/sample/

# Ingest from URLs
python scripts/ingest.py --url https://docs.example.com/faq https://docs.example.com/guide

# Ingest from a database
python scripts/ingest.py --db sqlite:///data/mydb.sqlite --table faq_entries

# Reset and re-ingest
python scripts/ingest.py --source ./data/sample/ --reset
```

### 4. Run

**Option A — Streamlit UI (Recommended for demo)**
```bash
python main.py --ui
# Open http://localhost:8501
```

**Option B — FastAPI Server (For integration)**
```bash
python main.py
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

**Option C — Docker**
```bash
docker-compose up --build
# UI:  http://localhost:8501
# API: http://localhost:8000
```

---

## 🖥️ Usage

### Chat via Streamlit UI

The web interface provides:
- 💬 Chat with your documents in a familiar interface
- 📤 Upload files directly from the sidebar (PDF, TXT, CSV, DOCX, HTML, Markdown)
- 🌐 Add web pages by URL
- ⚙️ Switch retrieval strategies (similarity / MMR / rerank)
- 📎 View source citations for every answer

### Chat via API

```bash
# Ask a question
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the remote work policy?",
    "retrieval_strategy": "similarity"
  }'

# Upload a document
curl -X POST http://localhost:8000/api/ingest/file \
  -F "file=@./data/sample/company_faq.md"

# Ingest web pages
curl -X POST http://localhost:8000/api/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://docs.example.com/guide"]}'

# Check collection stats
curl http://localhost:8000/api/stats
```

### Multi-Turn Conversation

The chatbot supports follow-up questions with conversation memory:

```json
{
  "question": "What about for enterprise customers?",
  "chat_history": [
    {"role": "user", "content": "What are the support hours?"},
    {"role": "assistant", "content": "Support is available Monday-Friday 9AM-6PM EST for Business plan customers."}
  ]
}
```

---

## 📁 Project Structure

```
Private-Chatbot-RAG/
│
├── config/
│   └── settings.py          # Centralized configuration (Pydantic Settings)
│
├── src/
│   ├── ingestion/            # Document loading & processing
│   │   ├── pdf_loader.py     # PDF extraction (PyPDF2 + pdfplumber + OCR)
│   │   ├── web_loader.py     # Web scraping & crawling
│   │   ├── database_loader.py # SQL database ingestion
│   │   ├── file_loader.py    # TXT, CSV, HTML, DOCX loader
│   │   └── text_splitter.py  # Chunking with RecursiveCharacterTextSplitter
│   │
│   ├── embedding/
│   │   └── embedder.py       # Embedding factory (OpenAI / HuggingFace / Cohere / Ollama)
│   │
│   ├── vectorstore/
│   │   └── store.py          # Vector store manager (ChromaDB / Qdrant)
│   │
│   ├── retrieval/
│   │   └── retriever.py      # Retrieval strategies (Similarity / MMR / Rerank)
│   │
│   ├── llm/
│   │   └── chat.py           # LLM factory (OpenAI / Anthropic / Ollama)
│   │
│   ├── chains/
│   │   └── rag_chain.py      # RAG orchestration with conversation memory
│   │
│   ├── api/
│   │   └── routes.py         # FastAPI REST endpoints
│   │
│   └── ui/
│       └── app.py            # Streamlit chat interface
│
├── scripts/
│   └── ingest.py             # CLI ingestion tool
│
├── tests/
│   └── test_pipeline.py      # Unit & integration tests
│
├── data/
│   └── sample/               # Sample documents for testing
│
├── docs/
│   └── architecture.md       # Detailed architecture documentation
│
├── .env.example              # Environment variable template
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container image
├── docker-compose.yml        # Multi-service deployment
├── main.py                   # Application entry point
└── README.md
```

---

## ⚙️ Configuration

All settings are managed through environment variables (`.env` file). Key options:

### LLM Providers

| Provider | Model | Cost | Setup |
|---|---|---|---|
| **OpenAI** | `gpt-4o-mini` | ~$0.15/1M tokens | Set `OPENAI_API_KEY` |
| **Anthropic** | `claude-3-5-sonnet` | ~$3/1M tokens | Set `ANTHROPIC_API_KEY` |
| **Ollama** | `llama3.2` | **Free (local)** | Install [Ollama](https://ollama.ai), run `ollama pull llama3.2` |

### Embedding Providers

| Provider | Model | Dimensions | Cost |
|---|---|---|---|
| **OpenAI** | `text-embedding-3-small` | 1536 | ~$0.02/1M tokens |
| **Ollama** | `nomic-embed-text` | 768 | **Free (local)** |
| **HuggingFace** | `all-MiniLM-L6-v2` | 384 | **Free (local)** |
| **Cohere** | `embed-english-v3.0` | 1024 | Free tier available |

### Fully Free/Local Setup (No API Keys Needed)

```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
VECTORSTORE_PROVIDER=chroma
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html
```

---

## 🔍 How RAG Works (Technical Deep Dive)

### The Problem
LLMs are trained on public data and hallucinate when asked about private/domain-specific information. Fine-tuning is expensive and inflexible.

### The Solution: Retrieval-Augmented Generation

**Ingestion Phase (offline):**
1. **Load** documents from various sources (PDF, web, DB)
2. **Split** documents into overlapping chunks (~1000 chars each)
3. **Embed** each chunk into a dense vector using an embedding model
4. **Store** vectors in a vector database with metadata

**Query Phase (real-time):**
1. **Embed** the user's question using the same embedding model
2. **Retrieve** the top-K most similar chunks via vector similarity search
3. **Augment** the LLM prompt with retrieved context
4. **Generate** a grounded answer with source citations

### Why This Architecture?

- **No hallucination** — The LLM is instructed to answer ONLY from provided context
- **Source traceability** — Every answer includes citations to the original documents
- **Instant updates** — Add new documents without retraining; they're immediately searchable
- **Data privacy** — Documents never leave your infrastructure (especially with Ollama + ChromaDB)
- **Cost efficient** — Embedding is cheap; you only pay for LLM tokens on the retrieved context

---

## 🗺️ Roadmap

- [x] Multi-format document ingestion (PDF, Web, DB, Files)
- [x] Multiple embedding providers (OpenAI, HuggingFace, Cohere, Ollama)
- [x] Multiple LLM providers (including local/free Ollama)
- [x] Advanced retrieval (Similarity, MMR, Reranking)
- [x] Conversation memory for multi-turn chat
- [x] REST API with FastAPI
- [x] Streamlit chat UI
- [x] Docker deployment
- [ ] Streaming responses (SSE)
- [ ] Authentication & multi-user support
- [ ] Evaluation framework (RAGAS metrics)
- [ ] Hybrid search (dense + sparse/BM25)
- [ ] Document management UI (view, delete individual docs)
- [ ] Webhook notifications for ingestion events

---

## 🤝 Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) — LLM application framework
- [ChromaDB](https://github.com/chroma-core/chroma) — Open-source embedding database
- [Streamlit](https://streamlit.io) — Rapid UI development
- [FastAPI](https://fastapi.tiangolo.com) — Modern Python web framework
- [Ollama](https://ollama.ai) — Run LLMs locally

---

<p align="center">
  Built with ❤️ to demonstrate production-grade AI engineering
</p>
