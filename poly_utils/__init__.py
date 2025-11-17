"""
Poly Utils - Shared utilities for all poly-maker bots

This package provides common utilities used across all trading bots:
- LLM Client: Unified interface for multiple LLM providers
- News Feed: Multi-category news aggregator
- Logging: Structured logging for bots
- Proxy Config: Proxy management
- Google Utils: Google Sheets integration
"""

from poly_utils.llm_client import LLMClient
from poly_utils.news_feed import NewsFeed, NewsItem
from poly_utils.logging_utils import get_logger, BotLogger
from poly_utils.proxy_config import setup_proxy

__all__ = [
    'LLMClient',
    'NewsFeed',
    'NewsItem',
    'get_logger',
    'BotLogger',
    'setup_proxy',
]
