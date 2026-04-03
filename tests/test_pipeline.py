"""
Test Suite
Run: pytest tests/ -v
"""

import os
import sys
import tempfile

import pytest

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ═══════════════════════════════════════════════════════════════
# Text Splitter Tests
# ═══════════════════════════════════════════════════════════════
class TestTextSplitter:
    def test_split_creates_chunks(self):
        from langchain_core.documents import Document
        from src.ingestion.text_splitter import TextSplitter

        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        doc = Document(
            page_content="Hello world. " * 50,
            metadata={"source": "test"},
        )
        chunks = splitter.split([doc])
        assert len(chunks) > 1
        assert all(len(c.page_content) <= 120 for c in chunks)  # allow some flex

    def test_split_preserves_metadata(self):
        from langchain_core.documents import Document
        from src.ingestion.text_splitter import TextSplitter

        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        doc = Document(
            page_content="Testing metadata propagation. " * 20,
            metadata={"source": "test.pdf", "page": 1},
        )
        chunks = splitter.split([doc])
        for chunk in chunks:
            assert chunk.metadata["source"] == "test.pdf"
            assert chunk.metadata["page"] == 1
            assert "chunk_index" in chunk.metadata

    def test_split_text_returns_strings(self):
        from src.ingestion.text_splitter import TextSplitter

        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        pieces = splitter.split_text("A" * 200)
        assert all(isinstance(p, str) for p in pieces)
        assert len(pieces) > 1


# ═══════════════════════════════════════════════════════════════
# File Loader Tests
# ═══════════════════════════════════════════════════════════════
class TestFileLoader:
    def test_load_txt(self):
        from src.ingestion.file_loader import FileLoader

        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("Hello, this is a test document.")
            f.flush()
            loader = FileLoader()
            docs = loader.load(f.name)
            assert len(docs) == 1
            assert "test document" in docs[0].page_content
        os.unlink(f.name)

    def test_load_csv(self):
        from src.ingestion.file_loader import FileLoader

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("name,role\nAlice,Engineer\nBob,Designer\n")
            f.flush()
            loader = FileLoader()
            docs = loader.load(f.name)
            assert len(docs) == 2
            assert "Alice" in docs[0].page_content
        os.unlink(f.name)

    def test_unsupported_extension_raises(self):
        from src.ingestion.file_loader import FileLoader

        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            loader = FileLoader()
            with pytest.raises(ValueError, match="Unsupported"):
                loader.load(f.name)
        os.unlink(f.name)


# ═══════════════════════════════════════════════════════════════
# PDF Loader Tests
# ═══════════════════════════════════════════════════════════════
class TestPDFLoader:
    def test_missing_file_raises(self):
        from src.ingestion.pdf_loader import PDFLoader

        loader = PDFLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/file.pdf")

    def test_non_pdf_raises(self):
        from src.ingestion.pdf_loader import PDFLoader

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            loader = PDFLoader()
            with pytest.raises(ValueError, match="Not a PDF"):
                loader.load(f.name)
        os.unlink(f.name)


# ═══════════════════════════════════════════════════════════════
# Configuration Tests
# ═══════════════════════════════════════════════════════════════
class TestConfig:
    def test_default_settings(self):
        from config.settings import Settings

        s = Settings()
        assert s.chunk_size == 1000
        assert s.chunk_overlap == 200
        assert s.retrieval_top_k == 5

    def test_settings_types(self):
        from config.settings import Settings, EmbeddingProvider

        s = Settings()
        assert isinstance(s.embedding_provider, EmbeddingProvider)
        assert isinstance(s.chunk_size, int)


# ═══════════════════════════════════════════════════════════════
# Database Loader Tests
# ═══════════════════════════════════════════════════════════════
class TestDatabaseLoader:
    def test_load_sqlite_table(self):
        import sqlite3
        from src.ingestion.database_loader import DatabaseLoader

        db_path = tempfile.mktemp(suffix=".db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE faq (question TEXT, answer TEXT)")
        conn.execute(
            "INSERT INTO faq VALUES ('What is RAG?', 'Retrieval-Augmented Generation')"
        )
        conn.commit()
        conn.close()

        loader = DatabaseLoader(f"sqlite:///{db_path}")
        docs = loader.load_table("faq")
        assert len(docs) == 1
        assert "RAG" in docs[0].page_content
        os.unlink(db_path)

    def test_list_tables(self):
        import sqlite3
        from src.ingestion.database_loader import DatabaseLoader

        db_path = tempfile.mktemp(suffix=".db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE test_table (id INTEGER)")
        conn.commit()
        conn.close()

        loader = DatabaseLoader(f"sqlite:///{db_path}")
        tables = loader.list_tables()
        assert "test_table" in tables
        os.unlink(db_path)
