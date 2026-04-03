"""
Web Document Loader
Scrapes web pages and converts HTML content to clean text documents.
Supports single URLs, sitemaps, and recursive crawling.
"""

import logging
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from config.settings import settings

logger = logging.getLogger(__name__)


class WebLoader:
    """Load documents from web pages."""

    def __init__(self, timeout: Optional[int] = None):
        self.timeout = timeout or settings.web_scrape_timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (compatible; PrivateRAGBot/1.0; "
                    "+https://github.com/yourusername/private-rag-chatbot)"
                )
            }
        )

    def load_url(self, url: str) -> list[Document]:
        """Scrape a single URL and return documents."""
        logger.info(f"Scraping URL: {url}")
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Extract main content
        main = soup.find("main") or soup.find("article") or soup.find("body")
        text = main.get_text(separator="\n", strip=True) if main else ""

        if not text.strip():
            logger.warning(f"No text content extracted from {url}")
            return []

        # Extract metadata
        title = soup.find("title")
        description = soup.find("meta", attrs={"name": "description"})

        return [
            Document(
                page_content=text.strip(),
                metadata={
                    "source": url,
                    "domain": urlparse(url).netloc,
                    "title": title.string if title else "",
                    "description": (
                        description.get("content", "") if description else ""
                    ),
                    "loader": "web",
                    "content_length": len(text),
                },
            )
        ]

    def load_urls(self, urls: list[str]) -> list[Document]:
        """Scrape multiple URLs."""
        documents = []
        for url in urls:
            docs = self.load_url(url)
            documents.extend(docs)
        logger.info(f"Loaded {len(documents)} documents from {len(urls)} URLs")
        return documents

    def load_sitemap(self, sitemap_url: str, max_pages: int = 50) -> list[Document]:
        """Parse a sitemap XML and scrape discovered URLs."""
        logger.info(f"Loading sitemap: {sitemap_url}")
        try:
            response = self.session.get(sitemap_url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch sitemap: {e}")
            return []

        soup = BeautifulSoup(response.text, "xml")
        urls = [loc.text for loc in soup.find_all("loc")][:max_pages]
        logger.info(f"Found {len(urls)} URLs in sitemap (limited to {max_pages})")

        return self.load_urls(urls)

    def crawl(
        self, start_url: str, max_pages: int = 20, same_domain: bool = True
    ) -> list[Document]:
        """
        Simple recursive crawler.
        Follows links from the start page up to max_pages.
        """
        visited = set()
        to_visit = [start_url]
        documents = []
        base_domain = urlparse(start_url).netloc

        while to_visit and len(visited) < max_pages:
            url = to_visit.pop(0)
            if url in visited:
                continue

            visited.add(url)
            docs = self.load_url(url)
            documents.extend(docs)

            # Discover new links
            try:
                response = self.session.get(url, timeout=self.timeout)
                soup = BeautifulSoup(response.text, "html.parser")
                for link in soup.find_all("a", href=True):
                    href = urljoin(url, link["href"])
                    parsed = urlparse(href)
                    # Clean URL (remove fragments)
                    clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                    if (
                        clean_url not in visited
                        and (not same_domain or parsed.netloc == base_domain)
                        and parsed.scheme in ("http", "https")
                    ):
                        to_visit.append(clean_url)
            except Exception as e:
                logger.warning(f"Error discovering links from {url}: {e}")

        logger.info(f"Crawled {len(visited)} pages, extracted {len(documents)} docs")
        return documents
