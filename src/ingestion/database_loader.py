"""
Database Document Loader
Connects to SQL databases and converts rows/tables into documents
suitable for embedding and retrieval.
"""

import logging
from typing import Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DatabaseLoader:
    """Load documents from SQL databases (SQLite, PostgreSQL, MySQL)."""

    def __init__(self, connection_string: str):
        """
        Args:
            connection_string: SQLAlchemy-style connection string.
                Examples:
                  - sqlite:///data/mydb.sqlite
                  - postgresql://user:pass@localhost:5432/dbname
                  - mysql://user:pass@localhost:3306/dbname
        """
        self.connection_string = connection_string
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            from sqlalchemy import create_engine

            self._engine = create_engine(self.connection_string)
        return self._engine

    def load_table(
        self,
        table_name: str,
        columns: Optional[list[str]] = None,
        content_columns: Optional[list[str]] = None,
        metadata_columns: Optional[list[str]] = None,
        where_clause: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[Document]:
        """
        Load rows from a table as documents.

        Args:
            table_name:       Table to query.
            columns:          Columns to SELECT (default: all).
            content_columns:  Columns whose values form page_content.
                              If None, all non-metadata columns are used.
            metadata_columns: Columns stored as metadata.
            where_clause:     Optional SQL WHERE filter (no "WHERE" keyword).
            limit:            Max rows to load.
        """
        from sqlalchemy import text

        col_str = ", ".join(columns) if columns else "*"
        query = f"SELECT {col_str} FROM {table_name}"
        if where_clause:
            query += f" WHERE {where_clause}"
        if limit:
            query += f" LIMIT {limit}"

        logger.info(f"Executing: {query}")

        documents = []
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            col_names = list(result.keys())

            for row in result:
                row_dict = dict(zip(col_names, row))

                # Build page_content
                if content_columns:
                    content_parts = [
                        f"{col}: {row_dict[col]}"
                        for col in content_columns
                        if col in row_dict and row_dict[col]
                    ]
                else:
                    meta_set = set(metadata_columns or [])
                    content_parts = [
                        f"{col}: {val}"
                        for col, val in row_dict.items()
                        if col not in meta_set and val
                    ]

                page_content = "\n".join(content_parts)
                if not page_content.strip():
                    continue

                # Build metadata
                metadata = {
                    "source": f"db://{table_name}",
                    "table": table_name,
                    "loader": "database",
                }
                if metadata_columns:
                    for col in metadata_columns:
                        if col in row_dict:
                            metadata[col] = row_dict[col]

                documents.append(
                    Document(page_content=page_content, metadata=metadata)
                )

        logger.info(f"Loaded {len(documents)} documents from {table_name}")
        return documents

    def load_query(self, query: str, content_columns: Optional[list[str]] = None) -> list[Document]:
        """
        Run an arbitrary SELECT query and convert results to documents.
        """
        from sqlalchemy import text

        logger.info(f"Running custom query: {query[:100]}...")
        documents = []

        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            col_names = list(result.keys())

            for row in result:
                row_dict = dict(zip(col_names, row))

                if content_columns:
                    parts = [
                        f"{c}: {row_dict[c]}" for c in content_columns if row_dict.get(c)
                    ]
                else:
                    parts = [f"{c}: {v}" for c, v in row_dict.items() if v]

                page_content = "\n".join(parts)
                if page_content.strip():
                    documents.append(
                        Document(
                            page_content=page_content,
                            metadata={"source": "custom_query", "loader": "database"},
                        )
                    )

        logger.info(f"Custom query returned {len(documents)} documents")
        return documents

    def list_tables(self) -> list[str]:
        """List all tables in the database."""
        from sqlalchemy import inspect

        inspector = inspect(self.engine)
        return inspector.get_table_names()
