"""
Ingestion CLI Script
Bulk-ingest documents from a directory, URLs, or database into the vector store.

Usage:
    python scripts/ingest.py --source ./data/sample/
    python scripts/ingest.py --url https://example.com/docs
    python scripts/ingest.py --db sqlite:///data/mydb.sqlite --table faq
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.settings import settings
from src.embedding.embedder import EmbeddingFactory
from src.vectorstore.store import VectorStoreManager
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.file_loader import FileLoader
from src.ingestion.web_loader import WebLoader
from src.ingestion.database_loader import DatabaseLoader
from src.ingestion.text_splitter import TextSplitter

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def ingest_directory(directory: str, vector_store: VectorStoreManager, splitter: TextSplitter):
    """Ingest all supported files from a directory."""
    path = Path(directory)
    if not path.exists():
        logger.error(f"Directory not found: {directory}")
        return

    pdf_loader = PDFLoader()
    file_loader = FileLoader()
    all_documents = []

    for file_path in sorted(path.rglob("*")):
        if file_path.is_dir():
            continue
        ext = file_path.suffix.lower()
        if ext not in settings.supported_extensions:
            logger.info(f"Skipping unsupported: {file_path.name}")
            continue

        try:
            if ext == ".pdf":
                docs = pdf_loader.load(str(file_path))
            else:
                docs = file_loader.load(str(file_path))
            all_documents.extend(docs)
            logger.info(f"Loaded: {file_path.name} ({len(docs)} docs)")
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")

    if all_documents:
        chunks = splitter.split(all_documents)
        ids = vector_store.add_documents(chunks)
        logger.info(f"Ingested {len(all_documents)} documents → {len(chunks)} chunks → {len(ids)} stored")
    else:
        logger.warning("No documents found to ingest")


def ingest_urls(urls: list[str], vector_store: VectorStoreManager, splitter: TextSplitter):
    """Ingest web pages."""
    loader = WebLoader()
    documents = loader.load_urls(urls)
    if documents:
        chunks = splitter.split(documents)
        ids = vector_store.add_documents(chunks)
        logger.info(f"Ingested {len(urls)} URLs → {len(chunks)} chunks")


def ingest_database(
    connection_string: str,
    table: str,
    vector_store: VectorStoreManager,
    splitter: TextSplitter,
):
    """Ingest rows from a database table."""
    loader = DatabaseLoader(connection_string)
    documents = loader.load_table(table)
    if documents:
        chunks = splitter.split(documents)
        ids = vector_store.add_documents(chunks)
        logger.info(f"Ingested {len(documents)} rows → {len(chunks)} chunks")


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG knowledge base")
    parser.add_argument("--source", type=str, help="Directory path to ingest files from")
    parser.add_argument("--url", type=str, nargs="+", help="URLs to scrape and ingest")
    parser.add_argument("--db", type=str, help="Database connection string")
    parser.add_argument("--table", type=str, help="Database table name (used with --db)")
    parser.add_argument("--reset", action="store_true", help="Delete existing collection first")
    args = parser.parse_args()

    if not any([args.source, args.url, args.db]):
        parser.print_help()
        sys.exit(1)

    # Initialize pipeline
    embedding = EmbeddingFactory.create()
    vector_store = VectorStoreManager(embedding)
    splitter = TextSplitter()

    if args.reset:
        logger.info("Resetting vector store collection...")
        vector_store.delete_collection()
        # Reinitialize after deletion
        vector_store = VectorStoreManager(embedding)

    if args.source:
        ingest_directory(args.source, vector_store, splitter)

    if args.url:
        ingest_urls(args.url, vector_store, splitter)

    if args.db:
        if not args.table:
            logger.error("--table is required when using --db")
            sys.exit(1)
        ingest_database(args.db, args.table, vector_store, splitter)

    # Print final stats
    stats = vector_store.get_collection_stats()
    logger.info(f"Collection stats: {stats}")


if __name__ == "__main__":
    main()
