"""
Poly Utils - Shared utilities for all poly-maker bots

This package provides common utilities used across all trading bots:
- LLM Client: Unified interface for multiple LLM providers
- News Feed: Multi-category news aggregator
- Semantic Search: ChromaDB-based semantic news matching (optional)
- Logging: Structured logging for bots
- Proxy Config: Proxy management
- Google Utils: Google Sheets integration
"""

from poly_utils.llm_client import LLMClient
from poly_utils.news_feed import NewsFeed, NewsItem
from poly_utils.logging_utils import get_logger, BotLogger
from poly_utils.proxy_config import setup_proxy

# Optional semantic search (requires chromadb + sentence-transformers)
try:
    from poly_utils.semantic_search import (
        SemanticSearchEngine,
        SearchResult,
        MarketEmbeddingCache
    )
    SEMANTIC_SEARCH_AVAILABLE = True
except ImportError:
    SemanticSearchEngine = None
    SearchResult = None
    MarketEmbeddingCache = None
    SEMANTIC_SEARCH_AVAILABLE = False

# Optional next-news-api integration
try:
    from poly_utils.next_news_integration import NextNewsAPIManager
    NEXT_NEWS_API_AVAILABLE = True
except ImportError:
    NextNewsAPIManager = None
    NEXT_NEWS_API_AVAILABLE = False

__all__ = [
    'LLMClient',
    'NewsFeed',
    'NewsItem',
    'get_logger',
    'BotLogger',
    'setup_proxy',
    # Semantic search (optional)
    'SemanticSearchEngine',
    'SearchResult',
    'MarketEmbeddingCache',
    'SEMANTIC_SEARCH_AVAILABLE',
    # Next News API (optional)
    'NextNewsAPIManager',
    'NEXT_NEWS_API_AVAILABLE',
]
