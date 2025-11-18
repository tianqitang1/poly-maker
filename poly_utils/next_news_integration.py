"""
Next News API Integration for Poly-Maker

Manages a local next-news-api server and provides a unified interface to fetch
news from multiple sources (Google News, NewsAPI, etc.).

next-news-api: https://github.com/riad-azz/next-news-api

Features:
- Automatic server lifecycle management (start/stop)
- Health checks and auto-restart
- Unified API for multiple news sources
- Conversion to NewsItem format
- Thread-safe singleton pattern

Usage:
    from poly_utils.next_news_integration import NextNewsAPIManager

    config = {
        'enabled': True,
        'server_path': '/path/to/next-news-api',
        'port': 3000,
        'auto_start': True
    }

    manager = NextNewsAPIManager(config)

    # Fetch news (server auto-starts if needed)
    news_items = manager.fetch_news(
        source='google',
        query='cryptocurrency',
        max_results=10
    )

    # Cleanup (optional - auto-cleanup on exit)
    manager.stop_server()
"""

import os
import time
import subprocess
import requests
import atexit
import signal
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from threading import Lock

from poly_utils.logging_utils import get_logger
from poly_utils.news_feed import NewsItem

logger = get_logger('poly_utils.next_news_api')


class NextNewsAPIManager:
    """
    Manages next-news-api server lifecycle and provides news fetching interface.

    This class handles starting/stopping the Next.js server, health checks,
    and fetching news from various sources via the unified API.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls, config: Dict[str, Any]):
        """Singleton pattern to ensure only one server instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Next News API manager.

        Args:
            config: Configuration dict with server settings
        """
        # Only initialize once (singleton pattern)
        if self._initialized:
            return

        self.config = config
        self.enabled = config.get('enabled', False)
        self.server_path = config.get('server_path', '../next-news-api')
        self.port = config.get('port', 3000)
        self.host = config.get('host', 'localhost')
        self.auto_start = config.get('auto_start', True)
        self.startup_timeout = config.get('startup_timeout', 30)

        self.base_url = f"http://{self.host}:{self.port}"
        self.api_url = f"{self.base_url}/api/news"

        self.process = None
        self.is_running = False

        # Register cleanup on exit
        atexit.register(self.stop_server)

        logger.info(
            f"Initialized NextNewsAPIManager (enabled={self.enabled}, "
            f"port={self.port}, auto_start={self.auto_start})"
        )

        self._initialized = True

    def check_health(self) -> bool:
        """
        Check if the next-news-api server is healthy.

        Returns:
            True if server is responding, False otherwise
        """
        try:
            response = requests.get(self.base_url, timeout=5)
            return response.status_code == 200
        except:
            return False

    def start_server(self) -> bool:
        """
        Start the next-news-api server.

        Returns:
            True if server started successfully, False otherwise
        """
        if not self.enabled:
            logger.warning("Next News API is disabled in config")
            return False

        # Check if already running
        if self.check_health():
            logger.info("Next News API server is already running")
            self.is_running = True
            return True

        # Validate server path
        server_path = Path(self.server_path).expanduser().resolve()
        if not server_path.exists():
            logger.error(
                f"Next News API server path not found: {server_path}\n"
                f"Clone it with: git clone https://github.com/riad-azz/next-news-api {server_path}"
            )
            return False

        # Check if package.json exists
        package_json = server_path / 'package.json'
        if not package_json.exists():
            logger.error(f"package.json not found in {server_path}")
            return False

        # Check if dependencies installed
        node_modules = server_path / 'node_modules'
        if not node_modules.exists():
            logger.info("Installing dependencies (this may take a minute)...")
            try:
                subprocess.run(
                    ['npm', 'install'],
                    cwd=server_path,
                    check=True,
                    capture_output=True
                )
                logger.info("Dependencies installed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install dependencies: {e}")
                return False

        # Build if needed
        next_dir = server_path / '.next'
        if not next_dir.exists():
            logger.info("Building Next.js app (this may take a minute)...")
            try:
                subprocess.run(
                    ['npm', 'run', 'build'],
                    cwd=server_path,
                    check=True,
                    capture_output=True
                )
                logger.info("Build completed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to build: {e}")
                return False

        # Start the server
        logger.info(f"Starting Next News API server on port {self.port}...")

        try:
            # Set environment variables
            env = os.environ.copy()
            env['PORT'] = str(self.port)

            # Start server process
            self.process = subprocess.Popen(
                ['npm', 'run', 'start'],
                cwd=server_path,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group for easy cleanup
            )

            # Wait for server to be ready
            start_time = time.time()
            while time.time() - start_time < self.startup_timeout:
                if self.check_health():
                    logger.info(f"Next News API server started successfully on port {self.port}")
                    self.is_running = True
                    return True
                time.sleep(1)

            # Timeout - kill the process
            logger.error(f"Server failed to start within {self.startup_timeout}s")
            self.stop_server()
            return False

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False

    def stop_server(self):
        """Stop the next-news-api server."""
        if self.process:
            logger.info("Stopping Next News API server...")
            try:
                # Kill the entire process group
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=5)
            except:
                # Force kill if graceful shutdown fails
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except:
                    pass

            self.process = None
            self.is_running = False
            logger.info("Server stopped")

    def ensure_running(self) -> bool:
        """
        Ensure the server is running, starting it if necessary.

        Returns:
            True if server is running, False otherwise
        """
        if self.check_health():
            return True

        if self.auto_start:
            return self.start_server()

        logger.warning("Server is not running and auto_start is disabled")
        return False

    def fetch_news(
        self,
        source: str = 'google',
        query: Optional[str] = None,
        category: Optional[str] = None,
        max_results: int = 10,
        language: str = 'en',
        country: Optional[str] = None
    ) -> List[NewsItem]:
        """
        Fetch news from next-news-api.

        Args:
            source: News source ('google', 'newsapi', etc.)
            query: Search query (optional)
            category: News category (business, technology, sports, etc.)
            max_results: Maximum number of results
            language: Language code (default: 'en')
            country: Country code (e.g., 'us', 'uk')

        Returns:
            List of NewsItem objects
        """
        if not self.enabled:
            return []

        # Ensure server is running
        if not self.ensure_running():
            logger.error("Failed to start Next News API server")
            return []

        # Build request parameters
        params = {
            'source': source,
            'max_results': max_results,
            'language': language
        }

        if query:
            params['q'] = query
        if category:
            params['category'] = category
        if country:
            params['country'] = country

        # Fetch news
        try:
            response = requests.get(self.api_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Convert to NewsItem objects
            news_items = []
            articles = data.get('articles', [])

            for article in articles[:max_results]:
                try:
                    # Parse published date
                    published_str = article.get('publishedAt') or article.get('pubDate')
                    if published_str:
                        try:
                            published = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
                        except:
                            published = datetime.now()
                    else:
                        published = datetime.now()

                    # Determine category
                    item_category = category or self._detect_category(
                        article.get('title', '') + ' ' + article.get('description', '')
                    )

                    news_item = NewsItem(
                        title=article.get('title', ''),
                        summary=article.get('description', '') or article.get('content', '')[:200],
                        link=article.get('url', '') or article.get('link', ''),
                        published=published,
                        source=f"NextNews-{source.title()}",
                        category=item_category,
                        tags=[source, language]
                    )

                    news_items.append(news_item)

                except Exception as e:
                    logger.debug(f"Failed to parse article: {e}")
                    continue

            logger.info(f"Fetched {len(news_items)} articles from next-news-api ({source})")
            return news_items

        except requests.RequestException as e:
            logger.error(f"Failed to fetch news from next-news-api: {e}")
            return []
        except Exception as e:
            logger.error(f"Error parsing next-news-api response: {e}")
            return []

    def fetch_multiple_sources(
        self,
        sources: List[str],
        query: Optional[str] = None,
        category: Optional[str] = None,
        max_results_per_source: int = 10
    ) -> List[NewsItem]:
        """
        Fetch news from multiple sources and combine results.

        Args:
            sources: List of source names ('google', 'newsapi', etc.)
            query: Search query (optional)
            category: News category
            max_results_per_source: Max results per source

        Returns:
            Combined list of NewsItem objects
        """
        all_news = []

        for source in sources:
            news = self.fetch_news(
                source=source,
                query=query,
                category=category,
                max_results=max_results_per_source
            )
            all_news.extend(news)

        # Remove duplicates by link
        seen_links = set()
        unique_news = []
        for item in all_news:
            if item.link not in seen_links:
                unique_news.append(item)
                seen_links.add(item.link)

        # Sort by recency
        unique_news.sort(key=lambda x: x.published, reverse=True)

        return unique_news

    def _detect_category(self, text: str) -> Optional[str]:
        """Auto-detect category from text."""
        text_lower = text.lower()

        # Sports keywords
        if any(kw in text_lower for kw in ['nfl', 'nba', 'mlb', 'nhl', 'soccer', 'football',
                                            'basketball', 'baseball', 'hockey', 'game', 'team']):
            return 'sports'

        # Crypto keywords
        if any(kw in text_lower for kw in ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto',
                                            'blockchain', 'defi', 'nft']):
            return 'crypto'

        # Politics keywords
        if any(kw in text_lower for kw in ['election', 'president', 'congress', 'senate',
                                            'vote', 'political', 'democrat', 'republican']):
            return 'politics'

        # Tech keywords
        if any(kw in text_lower for kw in ['ai', 'technology', 'tech', 'software', 'startup',
                                            'google', 'apple', 'microsoft']):
            return 'technology'

        return 'general'

    def get_available_sources(self) -> List[str]:
        """
        Get list of available news sources.

        Returns:
            List of source names (short codes)
        """
        # API aggregator sources
        api_sources = ['google', 'newsapi', 'newsdata']

        # Built-in RSS feed sources (from src/lib/news/constants.ts)
        rss_sources = [
            # International
            'INTER-YN',    # Yahoo News
            'INTER-LH',    # Life Hacker

            # United States
            'US-NYT',      # New York Times
            'US-CNNN',     # CNN News
            'US-HP',       # Huffington Post
            'US-FN',       # Fox News
            'US-R',        # Reuters
            'US-P',        # Politico
            'US-LAT',      # Los Angeles Times

            # Australia
            'AU-SMHLN',    # Sydney Morning Herald - Latest News
            'AU-ABCN',     # ABC News
            'AU-TALN',     # The Age - Latest News
            'AU-PN',       # PerthNow
            'AU-TCTLN',    # The Canberra Times - Local News
            'AU-BTLN',     # Brisbane Times - Latest News
            'AU-IA',       # Independent Australia
            'AU-BNLH',     # Business News - Latest Headlines
            'AU-ID',       # InDaily
            'AU-C',        # Crikey
            'AU-MW',       # Michael West

            # Canada
            'CA-CBCN',     # CBC News
            'CA-CTVN',     # CTV News
            'CA-FP',       # Financial Post
            'CA-NP',       # National Post
            'CA-OC',       # Ottawa Citizen
            'CA-TP',       # The Province
            'CA-TST',      # Toronto Star
            'CA-TSU',      # Toronto Sun

            # Germany
            'DE-ZO',       # ZEIT ONLINE
            'DE-FO',       # FOCUS Online
            'DE-DW',       # Deutsche Welle
        ]

        return api_sources + rss_sources

    def get_status(self) -> Dict[str, Any]:
        """
        Get server status information.

        Returns:
            Dict with status info
        """
        return {
            'enabled': self.enabled,
            'running': self.check_health(),
            'port': self.port,
            'base_url': self.base_url,
            'process_id': self.process.pid if self.process else None,
            'available_sources': self.get_available_sources()
        }
