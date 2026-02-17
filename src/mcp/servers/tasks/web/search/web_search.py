"""
# Copyright 2025 Rowel Atienza. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

Web Search Tool

Provides web search using Ollama's web search API with DuckDuckGo fallback.
See: https://ollama.com/blog/web-search

Features:
- Primary search via Ollama web search API
- Automatic fallback to DuckDuckGo if Ollama fails
- Content cleaning and truncation
- Configurable result limits
- Retry logic for transient failures
"""

import json
import re
import html
import logging
from typing import Optional

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("ollama not installed - will use DuckDuckGo only")

# Try to import ddgs for fallback
try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    logger.warning("ddgs not installed - fallback search unavailable")


class WebSearch:
    """
    Web search tool with Ollama primary and DuckDuckGo fallback.

    Usage:
        search = WebSearch(max_results=5, max_content_length=2000)
        results = search.search("Python programming history")
    """

    def __init__(
        self,
        max_results: int = 5,
        max_content_length: int = 2000,
        timeout: float = 10.0,
        use_fallback: bool = True
    ):
        """
        Initialize the web search tool.

        Args:
            max_results: Maximum number of results to return (default: 5)
            max_content_length: Maximum characters per result content (default: 2000)
            timeout: Search timeout in seconds (default: 10.0)
            use_fallback: Whether to fallback to DuckDuckGo if Ollama fails (default: True)
        """
        self.max_results = max_results
        self.max_content_length = max_content_length
        self.timeout = timeout
        self.use_fallback = use_fallback

    def _clean_content(self, content: str) -> str:
        """Clean and normalize content from search results."""
        if not content:
            return ""

        # Remove mismatched/broken tags
        content = re.sub(r'\[/.*?\]', '', content)
        content = re.sub(r'\[.*?\]', '', content)

        # Decode HTML entities
        content = html.unescape(content)

        # Remove HTML tags
        content = re.sub(r'<[^>]+>', '', content)

        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()

        # Truncate if needed
        if len(content) > self.max_content_length:
            content = content[:self.max_content_length].rsplit(' ', 1)[0] + "..."

        return content

    def _search_ollama(self, query: str) -> Optional[list]:
        """Search using Ollama web search API."""
        if not OLLAMA_AVAILABLE:
            return None

        try:
            response = ollama.web_search(query)

            if not response or 'results' not in response:
                logger.warning("Ollama returned empty response")
                return None

            results = []
            for result in response['results'][:self.max_results]:
                results.append({
                    'title': result.get('title', 'No title'),
                    'url': result.get('url', ''),
                    'content': self._clean_content(result.get('content', '')),
                    'source': 'ollama'
                })

            return results if results else None

        except Exception as e:
            logger.warning(f"Ollama search failed: {str(e)}")
            return None

    def _search_ddgs(self, query: str) -> Optional[list]:
        """Search using DuckDuckGo as fallback."""
        if not DDGS_AVAILABLE:
            return None

        try:
            ddgs = DDGS(timeout=self.timeout)
            response = ddgs.text(query, max_results=self.max_results)

            results = []
            for result in response:
                results.append({
                    'title': result.get('title', 'No title'),
                    'url': result.get('href', result.get('link', '')),
                    'content': self._clean_content(result.get('body', '')),
                    'source': 'duckduckgo'
                })

            return results if results else None

        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {str(e)}")
            return None

    def search(self, query: str) -> str:
        """
        Search the web for the given query.

        Tries Ollama first, falls back to DuckDuckGo if configured.

        Args:
            query: The search query string

        Returns:
            JSON string with search results or error message
        """
        if not query or not query.strip():
            return json.dumps({"error": "Empty search query"})

        query = query.strip()

        # Try Ollama first
        results = self._search_ollama(query)

        # Fallback to DuckDuckGo if needed
        if results is None and self.use_fallback:
            logger.info("Falling back to DuckDuckGo search")
            results = self._search_ddgs(query)

        # Return results or error
        if results:
            return json.dumps(results, indent=2, ensure_ascii=False)
        else:
            return json.dumps({
                "error": "Search failed - no results from any provider",
                "query": query,
                "providers_tried": ["ollama", "duckduckgo"] if self.use_fallback else ["ollama"]
            })

    def search_with_metadata(self, query: str) -> dict:
        """
        Search and return results with metadata (non-JSON).

        Args:
            query: The search query string

        Returns:
            Dictionary with results and metadata
        """
        if not query or not query.strip():
            return {"error": "Empty search query", "results": [], "count": 0}

        query = query.strip()
        source_used = None

        # Try Ollama first
        results = self._search_ollama(query)
        if results:
            source_used = "ollama"

        # Fallback to DuckDuckGo if needed
        if results is None and self.use_fallback:
            results = self._search_ddgs(query)
            if results:
                source_used = "duckduckgo"

        return {
            "query": query,
            "results": results or [],
            "count": len(results) if results else 0,
            "source": source_used,
            "success": results is not None
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Web Search Tool")
    parser.add_argument("--query", type=str, default="Python programming language history",
                        help="Search query")
    parser.add_argument("--max-results", type=int, default=5,
                        help="Maximum number of results")
    parser.add_argument("--no-fallback", action="store_true",
                        help="Disable DuckDuckGo fallback")
    args = parser.parse_args()

    search_tool = WebSearch(
        max_results=args.max_results,
        use_fallback=not args.no_fallback
    )
    print(search_tool.search(args.query))
