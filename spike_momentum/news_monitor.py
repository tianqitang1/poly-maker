"""
Sports News Monitor

Fetches news from multiple free sources:
- ESPN RSS feeds (by sport)
- TheScore RSS feeds
- Google News RSS (with keywords)

No API keys required - all free RSS feeds.
"""

import feedparser
import time
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from urllib.parse import quote

from poly_utils.logging_utils import get_logger

logger = get_logger('spike_momentum.news')


class NewsItem:
    """Represents a single news item."""

    def __init__(
        self,
        title: str,
        summary: str,
        link: str,
        published: datetime,
        source: str,
        sport: Optional[str] = None
    ):
        self.title = title
        self.summary = summary
        self.link = link
        self.published = published
        self.source = source
        self.sport = sport

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'title': self.title,
            'summary': self.summary,
            'link': self.link,
            'published': self.published.isoformat(),
            'source': self.source,
            'sport': self.sport,
            'age_seconds': (datetime.now() - self.published).total_seconds()
        }

    def __repr__(self) -> str:
        age = (datetime.now() - self.published).total_seconds() / 60
        return f"<NewsItem: {self.title[:50]}... ({age:.0f}m ago)>"


class SportsNewsMonitor:
    """Monitor sports news from multiple free RSS sources."""

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

    # TheScore RSS feeds (simpler structure)
    THESCORE_RSS_FEEDS = {
        'nfl': 'https://www.thescore.com/rss/nfl',
        'nba': 'https://www.thescore.com/rss/nba',
        'mlb': 'https://www.thescore.com/rss/mlb',
        'nhl': 'https://www.thescore.com/rss/nhl',
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize news monitor.

        Args:
            config: News configuration from config.yaml
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.sources_config = config.get('sources', {})
        self.max_age = config.get('max_age', 3600)  # seconds

        # Cache to avoid re-processing same items
        self.seen_links = set()
        self.news_cache = []  # Recent news items

        logger.info(f"Initialized SportsNewsMonitor (enabled={self.enabled})")

    def fetch_news(self, max_items: int = 100) -> List[NewsItem]:
        """
        Fetch latest news from all configured sources.

        Args:
            max_items: Maximum news items to return

        Returns:
            List of NewsItem objects, sorted by recency
        """
        if not self.enabled:
            return []

        all_items = []

        # Fetch from ESPN
        if self.sources_config.get('espn_rss', {}).get('enabled', True):
            espn_leagues = self.sources_config.get('espn_rss', {}).get('leagues', ['nfl', 'nba'])
            all_items.extend(self._fetch_espn(espn_leagues))

        # Fetch from TheScore
        if self.sources_config.get('thescore_rss', {}).get('enabled', True):
            score_leagues = self.sources_config.get('thescore_rss', {}).get('leagues', ['nfl', 'nba'])
            all_items.extend(self._fetch_thescore(score_leagues))

        # Fetch from Google News (if keywords provided)
        if self.sources_config.get('google_news_rss', {}).get('enabled', False):
            # For now, skip Google News - we'll add keyword-based search later
            pass

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

        logger.info(f"Fetched {len(unique_items)} news items (showing {min(len(unique_items), max_items)})")

        return self.news_cache[:max_items]

    def _fetch_espn(self, leagues: List[str]) -> List[NewsItem]:
        """Fetch news from ESPN RSS feeds."""
        items = []

        for league in leagues:
            if league not in self.ESPN_RSS_FEEDS:
                logger.warning(f"Unknown ESPN league: {league}")
                continue

            url = self.ESPN_RSS_FEEDS[league]

            try:
                feed = feedparser.parse(url)

                for entry in feed.entries:
                    # Parse published date
                    published = self._parse_date(entry.get('published_parsed'))

                    # Skip if already seen
                    link = entry.get('link', '')
                    if link in self.seen_links:
                        continue

                    item = NewsItem(
                        title=entry.get('title', ''),
                        summary=entry.get('summary', entry.get('description', '')),
                        link=link,
                        published=published,
                        source='ESPN',
                        sport=league.upper()
                    )

                    items.append(item)
                    self.seen_links.add(link)

                logger.debug(f"Fetched {len(feed.entries)} items from ESPN {league.upper()}")

            except Exception as e:
                logger.error(f"Error fetching ESPN {league}: {e}")

        return items

    def _fetch_thescore(self, leagues: List[str]) -> List[NewsItem]:
        """Fetch news from TheScore RSS feeds."""
        items = []

        for league in leagues:
            if league not in self.THESCORE_RSS_FEEDS:
                logger.warning(f"Unknown TheScore league: {league}")
                continue

            url = self.THESCORE_RSS_FEEDS[league]

            try:
                feed = feedparser.parse(url)

                for entry in feed.entries:
                    # Parse published date
                    published = self._parse_date(entry.get('published_parsed'))

                    # Skip if already seen
                    link = entry.get('link', '')
                    if link in self.seen_links:
                        continue

                    item = NewsItem(
                        title=entry.get('title', ''),
                        summary=entry.get('summary', entry.get('description', '')),
                        link=link,
                        published=published,
                        source='TheScore',
                        sport=league.upper()
                    )

                    items.append(item)
                    self.seen_links.add(link)

                logger.debug(f"Fetched {len(feed.entries)} items from TheScore {league.upper()}")

            except Exception as e:
                logger.error(f"Error fetching TheScore {league}: {e}")

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

    def get_recent_news(self, max_age_seconds: Optional[int] = None) -> List[NewsItem]:
        """
        Get recent news from cache.

        Args:
            max_age_seconds: Override default max age

        Returns:
            List of recent NewsItem objects
        """
        if max_age_seconds is None:
            max_age_seconds = self.max_age

        cutoff_time = datetime.now() - timedelta(seconds=max_age_seconds)

        return [
            item for item in self.news_cache
            if item.published > cutoff_time
        ]

    def search_news(self, keywords: List[str], max_results: int = 10) -> List[NewsItem]:
        """
        Search cached news for keywords.

        Args:
            keywords: List of keywords to search for
            max_results: Maximum results to return

        Returns:
            List of matching NewsItem objects
        """
        results = []

        # Build regex pattern (case-insensitive)
        pattern = '|'.join([re.escape(kw) for kw in keywords])
        regex = re.compile(pattern, re.IGNORECASE)

        for item in self.news_cache:
            # Search in title and summary
            if regex.search(item.title) or regex.search(item.summary):
                results.append(item)

                if len(results) >= max_results:
                    break

        return results

    def match_to_market(self, market_question: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Find news items relevant to a market question.

        Args:
            market_question: The Polymarket question
            max_results: Max news items to return

        Returns:
            List of dicts with news items and relevance scores
        """
        # Extract potential keywords from market question
        keywords = self._extract_keywords(market_question)

        if not keywords:
            return []

        # Search for matching news
        matching_news = self.search_news(keywords, max_results=max_results)

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

        return results

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract potential keywords from text."""
        # Remove common words and extract meaningful terms
        stop_words = {'will', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}

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

    def clear_cache(self):
        """Clear the news cache."""
        self.news_cache = []
        self.seen_links = set()
        logger.info("Cleared news cache")


def test_news_monitor():
    """Test the news monitor."""
    config = {
        'enabled': True,
        'max_age': 3600,
        'sources': {
            'espn_rss': {
                'enabled': True,
                'leagues': ['nfl', 'nba']
            },
            'thescore_rss': {
                'enabled': True,
                'leagues': ['nfl', 'nba']
            }
        }
    }

    monitor = SportsNewsMonitor(config)

    print("Fetching news...")
    news_items = monitor.fetch_news(max_items=10)

    print(f"\nFound {len(news_items)} news items:\n")
    for i, item in enumerate(news_items[:5], 1):
        print(f"{i}. [{item.source} - {item.sport}] {item.title}")
        print(f"   Published: {item.published.strftime('%Y-%m-%d %H:%M')}")
        print(f"   Summary: {item.summary[:100]}...")
        print()

    # Test market matching
    print("\nTesting market matching...")
    market_question = "Will the Lakers beat the Warriors tonight?"
    matches = monitor.match_to_market(market_question, max_results=3)

    print(f"\nMatches for: '{market_question}'")
    for match in matches:
        item = match['news']
        print(f"- {item.title} (score: {match['relevance_score']:.2f})")
        print(f"  Keywords: {', '.join(match['matched_keywords'])}")


if __name__ == '__main__':
    test_news_monitor()
