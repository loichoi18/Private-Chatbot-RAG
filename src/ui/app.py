"""
Streamlit Chat Interface
A clean, interactive UI for the Private RAG Chatbot.
"""

import streamlit as st
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.embedding.embedder import EmbeddingFactory
from src.vectorstore.store import VectorStoreManager
from src.retrieval.retriever import Retriever
from src.llm.chat import LLMFactory
from src.chains.rag_chain import RAGChain
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.file_loader import FileLoader
from src.ingestion.web_loader import WebLoader
from src.ingestion.text_splitter import TextSplitter

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Private RAG Chatbot",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Initialize Pipeline (cached) ────────────────────────────
@st.cache_resource
def init_pipeline():
    embedding = EmbeddingFactory.create()
    vector_store = VectorStoreManager(embedding)
    retriever = Retriever(vector_store)
    llm = LLMFactory.create()
    rag_chain = RAGChain(retriever, llm)
    splitter = TextSplitter()
    return vector_store, retriever, rag_chain, splitter


vector_store, retriever, rag_chain, splitter = init_pipeline()


# ── Session State ────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Sidebar: Document Ingestion ──────────────────────────────
with st.sidebar:
    st.title("📚 Knowledge Base")

    # Collection stats
    try:
        stats = vector_store.get_collection_stats()
        st.metric("Documents in Store", stats.get("count", 0))
    except Exception:
        st.metric("Documents in Store", "N/A")

    st.divider()

    # File upload
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["pdf", "txt", "md", "csv", "html", "docx"],
        accept_multiple_files=True,
        help="Supported: PDF, TXT, Markdown, CSV, HTML, DOCX",
    )

    if uploaded_files and st.button("📥 Ingest Files", type="primary"):
        for uploaded_file in uploaded_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                # Save temp file
                import tempfile

                ext = os.path.splitext(uploaded_file.name)[1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                try:
                    if ext == ".pdf":
                        docs = PDFLoader().load(tmp_path)
                    else:
                        docs = FileLoader().load(tmp_path)

                    for doc in docs:
                        doc.metadata["source"] = uploaded_file.name

                    chunks = splitter.split(docs)
                    vector_store.add_documents(chunks)
                    st.success(
                        f"✅ {uploaded_file.name}: {len(docs)} docs → {len(chunks)} chunks"
                    )
                except Exception as e:
                    st.error(f"❌ {uploaded_file.name}: {e}")
                finally:
                    os.unlink(tmp_path)

    st.divider()

    # URL ingestion
    st.subheader("Add Web Pages")
    url_input = st.text_area(
        "Enter URLs (one per line)",
        placeholder="https://example.com/docs\nhttps://example.com/faq",
    )

    if url_input and st.button("🌐 Ingest URLs"):
        urls = [u.strip() for u in url_input.strip().split("\n") if u.strip()]
        with st.spinner(f"Scraping {len(urls)} URLs..."):
            try:
                docs = WebLoader().load_urls(urls)
                chunks = splitter.split(docs)
                vector_store.add_documents(chunks)
                st.success(f"✅ Loaded {len(docs)} pages → {len(chunks)} chunks")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    st.divider()

    # Settings
    st.subheader("⚙️ Settings")
    strategy = st.selectbox(
        "Retrieval Strategy",
        ["similarity", "mmr", "rerank"],
        help="similarity=fast, mmr=diverse, rerank=most accurate",
    )

    if st.button("🗑️ Clear Knowledge Base"):
        try:
            vector_store.delete_collection()
            st.success("Knowledge base cleared")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

    if st.button("🔄 Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.rerun()


# ── Main Chat Area ───────────────────────────────────────────
st.title("🔒 Private RAG Chatbot")
st.caption("Ask questions about your documents. All data stays private.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📎 Sources"):
                for src in message["sources"]:
                    st.text(f"• {src.get('source', 'Unknown')}")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = rag_chain.query(
                    question=prompt,
                    chat_history=st.session_state.chat_history,
                    retrieval_strategy=strategy,
                )
                st.markdown(result["answer"])

                if result["sources"]:
                    with st.expander("📎 Sources"):
                        for src in result["sources"]:
                            source_name = src.get("filename") or src.get("source", "Unknown")
                            page = src.get("page")
                            score = src.get("relevance_score")
                            line = f"• {source_name}"
                            if page:
                                line += f" (p.{page})"
                            if score:
                                line += f" — score: {score}"
                            st.text(line)

                # Update history
                st.session_state.chat_history.append(
                    {"role": "user", "content": prompt}
                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": result["answer"]}
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    }
                )

            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Error: {e}"}
                )
