"""
PDF Document Loader
Extracts text from PDF files using multiple strategies:
  1. PyPDF2 for standard text-based PDFs
  2. pdfplumber for table-heavy PDFs
  3. OCR fallback via pytesseract for scanned documents
"""

import logging
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class PDFLoader:
    """Load and extract text from PDF documents."""

    def __init__(self, use_ocr_fallback: bool = False):
        self.use_ocr_fallback = use_ocr_fallback

    def load(self, file_path: str) -> list[Document]:
        """
        Load a PDF and return a list of Documents (one per page).
        Automatically selects the best extraction strategy.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {file_path}")

        logger.info(f"Loading PDF: {path.name} ({path.stat().st_size / 1024:.1f} KB)")

        documents = self._extract_with_pypdf(path)

        # If extraction yields very little text, try pdfplumber
        total_text = sum(len(doc.page_content) for doc in documents)
        if total_text < 100:
            logger.info("PyPDF2 extracted minimal text, trying pdfplumber...")
            documents = self._extract_with_pdfplumber(path)

        # OCR fallback for scanned documents
        total_text = sum(len(doc.page_content) for doc in documents)
        if total_text < 100 and self.use_ocr_fallback:
            logger.info("Attempting OCR fallback...")
            documents = self._extract_with_ocr(path)

        logger.info(
            f"Extracted {len(documents)} pages, "
            f"{sum(len(d.page_content) for d in documents)} chars total"
        )
        return documents

    def _extract_with_pypdf(self, path: Path) -> list[Document]:
        from PyPDF2 import PdfReader

        reader = PdfReader(str(path))
        documents = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(
                        page_content=text.strip(),
                        metadata={
                            "source": str(path),
                            "filename": path.name,
                            "page": i + 1,
                            "total_pages": len(reader.pages),
                            "loader": "pypdf2",
                        },
                    )
                )
        return documents

    def _extract_with_pdfplumber(self, path: Path) -> list[Document]:
        import pdfplumber

        documents = []
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                # Also extract tables as formatted text
                tables = page.extract_tables()
                table_text = ""
                for table in tables:
                    for row in table:
                        cleaned = [str(cell) if cell else "" for cell in row]
                        table_text += " | ".join(cleaned) + "\n"

                combined = f"{text}\n\n{table_text}".strip()
                if combined:
                    documents.append(
                        Document(
                            page_content=combined,
                            metadata={
                                "source": str(path),
                                "filename": path.name,
                                "page": i + 1,
                                "total_pages": len(pdf.pages),
                                "has_tables": len(tables) > 0,
                                "loader": "pdfplumber",
                            },
                        )
                    )
        return documents

    def _extract_with_ocr(self, path: Path) -> list[Document]:
        """OCR fallback using pdf2image + pytesseract."""
        try:
            from pdf2image import convert_from_path
            import pytesseract
        except ImportError:
            logger.warning("OCR dependencies not installed (pdf2image, pytesseract)")
            return []

        images = convert_from_path(str(path))
        documents = []
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            if text.strip():
                documents.append(
                    Document(
                        page_content=text.strip(),
                        metadata={
                            "source": str(path),
                            "filename": path.name,
                            "page": i + 1,
                            "total_pages": len(images),
                            "loader": "ocr",
                        },
                    )
                )
        return documents
