"""
Generalized News Feed Interface for Poly-Maker

Provides a unified interface for fetching news from multiple sources across
different categories (sports, politics, crypto, general news, etc.).

Supports:
- RSS feeds (ESPN, TheScore, Google News, CoinDesk, etc.)
- API-based news sources (extensible)
- Smart caching and deduplication
- Market relevance matching
- Category-specific filtering

Usage:
    from poly_utils.news_feed import NewsFeed

    config = {
        'enabled': True,
        'categories': ['sports', 'crypto'],
        'sources': {
            'espn_rss': {'enabled': True, 'leagues': ['nfl', 'nba']},
            'coindesk_rss': {'enabled': True}
        }
    }

    feed = NewsFeed(config)
    news_items = feed.fetch_news(category='sports')
    relevant_news = feed.match_to_market("Will Bitcoin reach $100k?")
"""

import feedparser
import time
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from urllib.parse import quote
from dataclasses import dataclass, field

from poly_utils.logging_utils import get_logger

# Optional semantic search support
try:
    from poly_utils.semantic_search import SemanticSearchEngine, SearchResult
    SEMANTIC_SEARCH_AVAILABLE = True
except ImportError:
    SEMANTIC_SEARCH_AVAILABLE = False
    SemanticSearchEngine = None
    SearchResult = None

# Optional next-news-api support
try:
    from poly_utils.next_news_integration import NextNewsAPIManager
    NEXT_NEWS_API_AVAILABLE = True
except ImportError:
    NEXT_NEWS_API_AVAILABLE = False
    NextNewsAPIManager = None

logger = get_logger('poly_utils.news_feed')


@dataclass
class NewsItem:
    """
    Represents a single news item from any source.
    """
    title: str
    summary: str
    link: str
    published: datetime
    source: str
    category: Optional[str] = None  # sports, crypto, politics, general, etc.
    subcategory: Optional[str] = None  # NFL, NBA, BTC, ETH, etc.
    tags: List[str] = field(default_factory=list)  # Additional tags for filtering

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'title': self.title,
            'summary': self.summary,
            'link': self.link,
            'published': self.published.isoformat(),
            'source': self.source,
            'category': self.category,
            'subcategory': self.subcategory,
            'tags': self.tags,
            'age_seconds': (datetime.now() - self.published).total_seconds()
        }

    def __repr__(self) -> str:
        age = (datetime.now() - self.published).total_seconds() / 60
        cat = f"{self.category}:{self.subcategory}" if self.subcategory else self.category
        return f"<NewsItem [{cat}] {self.title[:50]}... ({age:.0f}m ago)>"


class NewsFeed:
    """
    Generalized news feed aggregator supporting multiple categories and sources.
    """

    # ESPN RSS feeds by sport
    ESPN_RSS_FEEDS = {
        'nfl': 'https://www.espn.com/espn/rss/nfl/news',
        'nba': 'https://www.espn.com/espn/rss/nba/news',
        'mlb': 'https://www.espn.com/espn/rss/mlb/news',
        'nhl': 'https://www.espn.com/espn/rss/nhl/news',
        'soccer': 'https://www.espn.com/espn/rss/soccer/news',
        'ncaaf': 'https://www.espn.com/espn/rss/ncf/news',
        'ncaab': 'https://www.espn.com/espn/rss/ncb/news',
    }

    # TheScore RSS feeds
    THESCORE_RSS_FEEDS = {
        'nfl': 'https://www.thescore.com/rss/nfl',
        'nba': 'https://www.thescore.com/rss/nba',
        'mlb': 'https://www.thescore.com/rss/mlb',
        'nhl': 'https://www.thescore.com/rss/nhl',
    }

    # Crypto news RSS feeds
    CRYPTO_RSS_FEEDS = {
        'coindesk': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
        'cointelegraph': 'https://cointelegraph.com/rss',
        'bitcoin_magazine': 'https://bitcoinmagazine.com/feed',
        'decrypt': 'https://decrypt.co/feed',
    }

    # Political news RSS feeds
    POLITICS_RSS_FEEDS = {
        'politico': 'https://www.politico.com/rss/politics08.xml',
        'the_hill': 'https://thehill.com/feed/',
        'real_clear_politics': 'https://www.realclearpolitics.com/index.xml',
    }

    # General news RSS feeds
    GENERAL_RSS_FEEDS = {
        'reuters_world': 'https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best',
        'ap_news': 'https://apnews.com/apf-topnews',
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize news feed.

        Args:
            config: News configuration dict with sources and settings
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.sources_config = config.get('sources', {})
        self.max_age = config.get('max_age', 3600)  # seconds
        self.refresh_interval = config.get('refresh_interval', 60)  # seconds

        # Cache to avoid re-processing same items
        self.seen_links = set()
        self.news_cache = []  # Recent news items
        self.last_refresh = 0

        # Category filter (if specified, only fetch these categories)
        self.categories = config.get('categories', ['sports', 'crypto', 'politics', 'general'])

        # Initialize semantic search if enabled
        self.semantic_search = None
        semantic_config = config.get('semantic_search', {})
        if semantic_config.get('enabled', False):
            if SEMANTIC_SEARCH_AVAILABLE:
                try:
                    self.semantic_search = SemanticSearchEngine(semantic_config)
                    logger.info("Semantic search enabled")
                except Exception as e:
                    logger.warning(f"Failed to initialize semantic search: {e}")
            else:
                logger.warning(
                    "Semantic search requested but dependencies not available. "
                    "Install: pip install chromadb sentence-transformers"
                )

        # Initialize next-news-api if enabled
        self.next_news_api = None
        next_news_config = self.sources_config.get('next_news_api', {})
        if next_news_config.get('enabled', False):
            if NEXT_NEWS_API_AVAILABLE:
                try:
                    self.next_news_api = NextNewsAPIManager(next_news_config)
                    logger.info("Next News API integration enabled")
                except Exception as e:
                    logger.warning(f"Failed to initialize Next News API: {e}")
            else:
                logger.warning("Next News API requested but module not available")

        logger.info(
            f"Initialized NewsFeed (enabled={self.enabled}, categories={self.categories}, "
            f"semantic_search={self.semantic_search is not None}, "
            f"next_news_api={self.next_news_api is not None})"
        )

    def fetch_news(
        self,
        category: Optional[str] = None,
        max_items: int = 100,
        force_refresh: bool = False
    ) -> List[NewsItem]:
        """
        Fetch latest news from all configured sources.

        Args:
            category: Optional category filter (sports, crypto, politics, etc.)
            max_items: Maximum news items to return
            force_refresh: Force refresh even if within refresh interval

        Returns:
            List of NewsItem objects, sorted by recency
        """
        if not self.enabled:
            return []

        # Check if we need to refresh
        current_time = time.time()
        if not force_refresh and (current_time - self.last_refresh) < self.refresh_interval:
            # Use cached news
            if category:
                cached = [item for item in self.news_cache if item.category == category]
                return cached[:max_items]
            return self.news_cache[:max_items]

        # Fetch new news
        all_items = []

        # Sports news
        if not category or category == 'sports':
            all_items.extend(self._fetch_sports_news())

        # Crypto news
        if not category or category == 'crypto':
            all_items.extend(self._fetch_crypto_news())

        # Political news
        if not category or category == 'politics':
            all_items.extend(self._fetch_politics_news())

        # General news
        if not category or category == 'general':
            all_items.extend(self._fetch_general_news())

        # Next News API (if enabled)
        if self.next_news_api:
            all_items.extend(self._fetch_next_news_api(category))

        # Filter by age
        cutoff_time = datetime.now() - timedelta(seconds=self.max_age)
        all_items = [item for item in all_items if item.published > cutoff_time]

        # Remove duplicates based on link
        unique_items = []
        seen_in_batch = set()
        for item in all_items:
            if item.link not in seen_in_batch:
                unique_items.append(item)
                seen_in_batch.add(item.link)

        # Sort by recency
        unique_items.sort(key=lambda x: x.published, reverse=True)

        # Update cache
        self.news_cache = unique_items[:max_items]
        self.last_refresh = current_time

        # Add to semantic search index if enabled
        if self.semantic_search and unique_items:
            try:
                self.semantic_search.add_news(unique_items)
            except Exception as e:
                logger.warning(f"Failed to add news to semantic index: {e}")

        logger.info(
            f"Fetched {len(unique_items)} news items "
            f"(category={category or 'all'}, showing {min(len(unique_items), max_items)})"
        )

        return self.news_cache[:max_items]

    def _fetch_sports_news(self) -> List[NewsItem]:
        """Fetch sports news from ESPN and TheScore."""
        items = []

        # ESPN
        if self.sources_config.get('espn_rss', {}).get('enabled', True):
            espn_leagues = self.sources_config.get('espn_rss', {}).get('leagues', ['nfl', 'nba'])
            items.extend(self._fetch_rss_feeds(
                feeds=self.ESPN_RSS_FEEDS,
                leagues=espn_leagues,
                source='ESPN',
                category='sports'
            ))

        # TheScore
        if self.sources_config.get('thescore_rss', {}).get('enabled', True):
            score_leagues = self.sources_config.get('thescore_rss', {}).get('leagues', ['nfl', 'nba'])
            items.extend(self._fetch_rss_feeds(
                feeds=self.THESCORE_RSS_FEEDS,
                leagues=score_leagues,
                source='TheScore',
                category='sports'
            ))

        return items

    def _fetch_crypto_news(self) -> List[NewsItem]:
        """Fetch crypto news from multiple sources."""
        items = []

        if self.sources_config.get('coindesk_rss', {}).get('enabled', False):
            items.extend(self._fetch_single_rss(
                url=self.CRYPTO_RSS_FEEDS['coindesk'],
                source='CoinDesk',
                category='crypto'
            ))

        if self.sources_config.get('cointelegraph_rss', {}).get('enabled', False):
            items.extend(self._fetch_single_rss(
                url=self.CRYPTO_RSS_FEEDS['cointelegraph'],
                source='CoinTelegraph',
                category='crypto'
            ))

        if self.sources_config.get('bitcoin_magazine_rss', {}).get('enabled', False):
            items.extend(self._fetch_single_rss(
                url=self.CRYPTO_RSS_FEEDS['bitcoin_magazine'],
                source='Bitcoin Magazine',
                category='crypto'
            ))

        if self.sources_config.get('decrypt_rss', {}).get('enabled', False):
            items.extend(self._fetch_single_rss(
                url=self.CRYPTO_RSS_FEEDS['decrypt'],
                source='Decrypt',
                category='crypto'
            ))

        return items

    def _fetch_politics_news(self) -> List[NewsItem]:
        """Fetch political news from multiple sources."""
        items = []

        if self.sources_config.get('politico_rss', {}).get('enabled', False):
            items.extend(self._fetch_single_rss(
                url=self.POLITICS_RSS_FEEDS['politico'],
                source='Politico',
                category='politics'
            ))

        if self.sources_config.get('the_hill_rss', {}).get('enabled', False):
            items.extend(self._fetch_single_rss(
                url=self.POLITICS_RSS_FEEDS['the_hill'],
                source='The Hill',
                category='politics'
            ))

        if self.sources_config.get('real_clear_politics_rss', {}).get('enabled', False):
            items.extend(self._fetch_single_rss(
                url=self.POLITICS_RSS_FEEDS['real_clear_politics'],
                source='RealClearPolitics',
                category='politics'
            ))

        return items

    def _fetch_general_news(self) -> List[NewsItem]:
        """Fetch general news from multiple sources."""
        items = []

        if self.sources_config.get('reuters_rss', {}).get('enabled', False):
            items.extend(self._fetch_single_rss(
                url=self.GENERAL_RSS_FEEDS['reuters_world'],
                source='Reuters',
                category='general'
            ))

        if self.sources_config.get('ap_news_rss', {}).get('enabled', False):
            items.extend(self._fetch_single_rss(
                url=self.GENERAL_RSS_FEEDS['ap_news'],
                source='AP News',
                category='general'
            ))

        return items

    def _fetch_next_news_api(self, category: Optional[str] = None) -> List[NewsItem]:
        """Fetch news from next-news-api."""
        if not self.next_news_api:
            return []

        items = []
        config = self.sources_config.get('next_news_api', {})

        # Get configured sources (default to google)
        sources = config.get('sources', ['google'])

        # Map our categories to next-news-api categories
        category_map = {
            'sports': 'sports',
            'crypto': 'technology',  # Closest match
            'politics': 'politics',
            'general': 'general',
            'technology': 'technology'
        }

        next_category = category_map.get(category, category) if category else None

        # Get query if specified
        query = config.get('query')

        # Fetch from configured sources
        try:
            all_items = self.next_news_api.fetch_multiple_sources(
                sources=sources,
                query=query,
                category=next_category,
                max_results_per_source=config.get('max_results_per_source', 10)
            )

            # Filter by category if specified
            if category:
                all_items = [item for item in all_items if item.category == category]

            items.extend(all_items)

        except Exception as e:
            logger.error(f"Failed to fetch from next-news-api: {e}")

        return items

    def _fetch_rss_feeds(
        self,
        feeds: Dict[str, str],
        leagues: List[str],
        source: str,
        category: str
    ) -> List[NewsItem]:
        """Fetch from multiple RSS feeds (e.g., sports leagues)."""
        items = []

        for league in leagues:
            if league not in feeds:
                logger.warning(f"Unknown league/feed: {league} for {source}")
                continue

            url = feeds[league]
            items.extend(self._fetch_single_rss(
                url=url,
                source=source,
                category=category,
                subcategory=league.upper()
            ))

        return items

    def _fetch_single_rss(
        self,
        url: str,
        source: str,
        category: str,
        subcategory: Optional[str] = None
    ) -> List[NewsItem]:
        """Fetch from a single RSS feed."""
        items = []

        try:
            feed = feedparser.parse(url)

            for entry in feed.entries:
                # Parse published date
                published = self._parse_date(entry.get('published_parsed'))

                # Skip if already seen
                link = entry.get('link', '')
                if link in self.seen_links:
                    continue

                # Extract tags if available
                tags = []
                if hasattr(entry, 'tags'):
                    tags = [tag.term for tag in entry.tags]

                item = NewsItem(
                    title=entry.get('title', ''),
                    summary=entry.get('summary', entry.get('description', '')),
                    link=link,
                    published=published,
                    source=source,
                    category=category,
                    subcategory=subcategory,
                    tags=tags
                )

                items.append(item)
                self.seen_links.add(link)

            logger.debug(f"Fetched {len(feed.entries)} items from {source} ({category})")

        except Exception as e:
            logger.error(f"Error fetching RSS from {source} ({url}): {e}")

        return items

    def _parse_date(self, date_tuple) -> datetime:
        """Parse RSS date tuple to datetime."""
        if date_tuple:
            try:
                return datetime(*date_tuple[:6])
            except:
                pass

        # Default to now if parsing fails
        return datetime.now()

    def get_recent_news(
        self,
        category: Optional[str] = None,
        max_age_seconds: Optional[int] = None
    ) -> List[NewsItem]:
        """
        Get recent news from cache.

        Args:
            category: Optional category filter
            max_age_seconds: Override default max age

        Returns:
            List of recent NewsItem objects
        """
        if max_age_seconds is None:
            max_age_seconds = self.max_age

        cutoff_time = datetime.now() - timedelta(seconds=max_age_seconds)

        items = [
            item for item in self.news_cache
            if item.published > cutoff_time
        ]

        if category:
            items = [item for item in items if item.category == category]

        return items

    def search_news(
        self,
        keywords: List[str],
        category: Optional[str] = None,
        max_results: int = 10
    ) -> List[NewsItem]:
        """
        Search cached news for keywords.

        Args:
            keywords: List of keywords to search for
            category: Optional category filter
            max_results: Maximum results to return

        Returns:
            List of matching NewsItem objects
        """
        results = []

        # Build regex pattern (case-insensitive)
        pattern = '|'.join([re.escape(kw) for kw in keywords])
        regex = re.compile(pattern, re.IGNORECASE)

        # Filter by category first if specified
        search_items = self.news_cache
        if category:
            search_items = [item for item in search_items if item.category == category]

        for item in search_items:
            # Search in title and summary
            if regex.search(item.title) or regex.search(item.summary):
                results.append(item)

                if len(results) >= max_results:
                    break

        return results

    def match_to_market(
        self,
        market_question: str,
        category: Optional[str] = None,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find news items relevant to a market question.

        Args:
            market_question: The Polymarket question
            category: Optional category filter (auto-detect if None)
            max_results: Max news items to return

        Returns:
            List of dicts with news items and relevance scores
        """
        # Auto-detect category if not specified
        if not category:
            category = self._detect_category(market_question)

        # Extract potential keywords from market question
        keywords = self._extract_keywords(market_question)

        if not keywords:
            return []

        # Search for matching news
        matching_news = self.search_news(keywords, category=category, max_results=max_results * 2)

        # Calculate relevance scores (simple keyword matching)
        results = []
        for item in matching_news:
            score = self._calculate_relevance(item, keywords)

            results.append({
                'news': item,
                'relevance_score': score,
                'matched_keywords': self._get_matched_keywords(item, keywords)
            })

        # Sort by relevance
        results.sort(key=lambda x: x['relevance_score'], reverse=True)

        return results[:max_results]

    def semantic_match_to_market(
        self,
        market_question: str,
        market_id: Optional[str] = None,
        category: Optional[str] = None,
        max_results: int = 5,
        min_similarity: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Find news items semantically similar to a market question using embeddings.

        This method uses semantic search (ChromaDB + embeddings) to find relevant news
        instead of simple keyword matching. Much more accurate than match_to_market().

        Args:
            market_question: The Polymarket question
            market_id: Optional market ID (enables embedding caching)
            category: Optional category filter (auto-detect if None)
            max_results: Max news items to return
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of dicts with news items and similarity scores
            Format: [{'news': NewsItem, 'similarity_score': float, 'matched_keywords': []}]

        Raises:
            RuntimeError: If semantic search is not enabled/available
        """
        if not self.semantic_search:
            raise RuntimeError(
                "Semantic search not enabled. Enable in config with:\n"
                "news:\n"
                "  semantic_search:\n"
                "    enabled: true\n"
                "    provider: sentence_transformer"
            )

        # Cache market embedding if market_id provided
        if market_id:
            try:
                self.semantic_search.cache_market(
                    question=market_question,
                    market_id=market_id,
                    metadata={'category': category}
                )
            except Exception as e:
                logger.debug(f"Failed to cache market embedding: {e}")

        # Search semantically
        try:
            search_results = self.semantic_search.search_news(
                query=market_question,
                market_id=market_id,
                max_results=max_results,
                min_similarity=min_similarity
            )
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            # Fall back to keyword matching
            logger.info("Falling back to keyword matching")
            return self.match_to_market(market_question, category, max_results)

        # Convert SearchResult objects to match the format of match_to_market()
        results = []
        for result in search_results:
            # Reconstruct NewsItem from metadata
            news_item = NewsItem(
                title=result.metadata['title'],
                summary=result.metadata['summary'],
                link=result.metadata['link'],
                published=datetime.fromisoformat(result.metadata['published']),
                source=result.metadata['source'],
                category=result.metadata.get('category'),
                subcategory=result.metadata.get('subcategory'),
                tags=result.metadata.get('tags', [])
            )

            results.append({
                'news': news_item,
                'relevance_score': result.similarity_score,  # Using same key for compatibility
                'matched_keywords': []  # Semantic search doesn't use keywords
            })

        return results

    def _detect_category(self, text: str) -> Optional[str]:
        """Auto-detect category from text."""
        text_lower = text.lower()

        # Sports keywords
        sports_keywords = ['nfl', 'nba', 'mlb', 'nhl', 'soccer', 'football', 'basketball',
                          'baseball', 'hockey', 'game', 'team', 'player', 'score']
        if any(kw in text_lower for kw in sports_keywords):
            return 'sports'

        # Crypto keywords
        crypto_keywords = ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'blockchain',
                          'defi', 'nft', 'token', 'coin']
        if any(kw in text_lower for kw in crypto_keywords):
            return 'crypto'

        # Politics keywords
        politics_keywords = ['election', 'president', 'congress', 'senate', 'vote',
                           'political', 'party', 'democrat', 'republican']
        if any(kw in text_lower for kw in politics_keywords):
            return 'politics'

        return None  # General category

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract potential keywords from text."""
        # Remove common words and extract meaningful terms
        stop_words = {'will', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'be', 'is'}

        # Tokenize and clean
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        # Also extract quoted terms and proper nouns (capitalized)
        quoted = re.findall(r'"([^"]+)"', text)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)

        keywords.extend([q.lower() for q in quoted])
        keywords.extend([pn.lower() for pn in proper_nouns])

        return list(set(keywords))  # Remove duplicates

    def _calculate_relevance(self, item: NewsItem, keywords: List[str]) -> float:
        """Calculate relevance score (0-1) for a news item."""
        # Combine title and summary for scoring
        text = (item.title + ' ' + item.summary).lower()

        # Count keyword matches
        matches = sum(1 for kw in keywords if kw in text)

        # Normalize by total keywords
        if not keywords:
            return 0.0

        score = matches / len(keywords)

        # Boost score for title matches
        title_matches = sum(1 for kw in keywords if kw in item.title.lower())
        if title_matches > 0:
            score += 0.2  # Bonus for title match

        # Cap at 1.0
        return min(score, 1.0)

    def _get_matched_keywords(self, item: NewsItem, keywords: List[str]) -> List[str]:
        """Get list of keywords that matched this item."""
        text = (item.title + ' ' + item.summary).lower()
        return [kw for kw in keywords if kw in text]

    def cleanup_closed_markets(self, closed_market_ids: List[str]):
        """
        Remove closed markets from semantic search cache.

        Args:
            closed_market_ids: List of market IDs that have closed
        """
        if self.semantic_search:
            try:
                self.semantic_search.cleanup_closed_markets(closed_market_ids)
            except Exception as e:
                logger.warning(f"Failed to cleanup closed markets: {e}")

    def cleanup_old_markets(self, max_age_hours: int = 168):
        """
        Remove old markets from semantic search cache.

        Args:
            max_age_hours: Maximum age in hours (default: 7 days)
        """
        if self.semantic_search:
            try:
                self.semantic_search.cleanup_old_markets(max_age_hours)
            except Exception as e:
                logger.warning(f"Failed to cleanup old markets: {e}")

    def clear_cache(self):
        """Clear the news cache and semantic search index."""
        self.news_cache = []
        self.seen_links = set()
        self.last_refresh = 0

        if self.semantic_search:
            try:
                self.semantic_search.clear_news()
            except Exception as e:
                logger.warning(f"Failed to clear semantic search index: {e}")

        logger.info("Cleared news cache")
