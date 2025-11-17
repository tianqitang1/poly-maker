"""
Market Scanner for Spike Momentum Bot

Scans Polymarket for sports markets and monitors them for price spikes.
Integrates with existing WebSocket infrastructure.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime

from poly_data.polymarket_client import PolymarketClient
from spike_momentum.spike_detector import SpikeDetector, Spike
from spike_momentum.news_monitor import SportsNewsMonitor
from spike_momentum.llm_analyzer import LLMAnalyzer
from poly_utils.logging_utils import get_logger

logger = get_logger('spike_momentum.scanner')


class MarketScanner:
    """Scans Polymarket for sports markets and monitors for spikes."""

    SPORTS_KEYWORDS = [
        'nfl', 'nba', 'mlb', 'nhl', 'mls',
        'premier league', 'champions league', 'world cup',
        'football', 'basketball', 'baseball', 'hockey', 'soccer',
        'game', 'match', 'win', 'score', 'playoff',
        'lakers', 'warriors', 'chiefs', 'bills', 'cowboys',
        'manchester', 'liverpool', 'barcelona', 'real madrid'
    ]

    def __init__(
        self,
        client: PolymarketClient,
        spike_detector: SpikeDetector,
        news_monitor: SportsNewsMonitor,
        llm_analyzer: Optional[LLMAnalyzer] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize market scanner.

        Args:
            client: Polymarket client
            spike_detector: Spike detector instance
            news_monitor: News monitor instance
            llm_analyzer: Optional LLM analyzer
            config: Optional configuration
        """
        self.client = client
        self.spike_detector = spike_detector
        self.news_monitor = news_monitor
        self.llm_analyzer = llm_analyzer
        self.config = config or {}

        # Market data
        self.sports_markets = {}  # market_id -> market_info
        self.market_questions = {}  # market_id -> question text

        logger.info("Initialized MarketScanner")

    def fetch_sports_markets(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch sports markets from Polymarket.

        Args:
            limit: Maximum markets to return

        Returns:
            List of market dictionaries
        """
        logger.info("Fetching markets from Polymarket...")

        try:
            # Fetch all markets (pagination handled by get_sampling_markets)
            cursor = ""
            all_markets = []

            while len(all_markets) < limit:
                try:
                    response = self.client.client.get_sampling_markets(next_cursor=cursor)
                    markets_data = response.get('data', [])

                    if not markets_data:
                        break

                    all_markets.extend(markets_data)

                    cursor = response.get('next_cursor')
                    if cursor is None:
                        break

                except Exception as e:
                    logger.error(f"Error fetching markets batch: {e}")
                    break

            logger.info(f"Fetched {len(all_markets)} total markets")

            # Filter for sports markets
            sports_markets = []
            for market in all_markets:
                if self._is_sports_market(market):
                    sports_markets.append(market)

            logger.info(f"Found {len(sports_markets)} sports markets")

            # Store market info
            for market in sports_markets[:limit]:
                condition_id = market.get('condition_id', '')
                question = market.get('question', '')

                # Get tokens for this market
                tokens = market.get('tokens', [])
                if len(tokens) >= 2:
                    yes_token = tokens[0].get('token_id', '')
                    no_token = tokens[1].get('token_id', '')

                    self.sports_markets[condition_id] = {
                        'condition_id': condition_id,
                        'question': question,
                        'yes_token': yes_token,
                        'no_token': no_token,
                        'market_slug': market.get('market_slug', ''),
                        'category': market.get('category', ''),
                        'end_date': market.get('end_date_iso', ''),
                    }

                    self.market_questions[condition_id] = question

            return sports_markets[:limit]

        except Exception as e:
            logger.error(f"Error fetching sports markets: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _is_sports_market(self, market: Dict[str, Any]) -> bool:
        """Check if market is sports-related."""
        question = market.get('question', '').lower()
        category = market.get('category', '').lower()
        tags = ' '.join(market.get('tags', [])).lower()

        # Check if any sports keyword is in question, category, or tags
        text_to_check = f"{question} {category} {tags}"

        for keyword in self.SPORTS_KEYWORDS:
            if keyword in text_to_check:
                return True

        return False

    def get_token_ids(self) -> List[str]:
        """Get list of token IDs to subscribe to."""
        token_ids = []

        for market_info in self.sports_markets.values():
            yes_token = market_info.get('yes_token')
            no_token = market_info.get('no_token')

            if yes_token:
                token_ids.append(yes_token)
            if no_token:
                token_ids.append(no_token)

        logger.info(f"Generated {len(token_ids)} token IDs to monitor")
        return token_ids

    async def monitor_markets(self):
        """
        Monitor markets via WebSocket and detect spikes.

        This is the main monitoring loop.
        """
        logger.info("Starting market monitoring...")

        # Fetch sports markets first
        sports_markets = self.fetch_sports_markets(limit=50)

        if not sports_markets:
            logger.error("No sports markets found to monitor")
            return

        # Get token IDs to subscribe to
        token_ids = self.get_token_ids()

        if not token_ids:
            logger.error("No token IDs to monitor")
            return

        logger.info(f"Monitoring {len(self.sports_markets)} markets ({len(token_ids)} tokens)")

        # Connect to WebSocket and process updates
        uri = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

        try:
            import websockets

            async with websockets.connect(uri, ping_interval=5, ping_timeout=None) as websocket:
                # Subscribe to markets
                message = {"assets_ids": token_ids}
                await websocket.send(json.dumps(message))

                logger.info(f"Subscribed to {len(token_ids)} token IDs")

                # Process incoming messages
                while True:
                    try:
                        message = await websocket.recv()
                        json_data = json.loads(message)

                        # Process market updates
                        if isinstance(json_data, dict):
                            await self._process_market_update([json_data])
                        elif isinstance(json_data, list):
                            await self._process_market_update(json_data)

                    except Exception as e:
                        logger.error(f"Error processing message: {e}")

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            import traceback
            traceback.print_exc()

    async def _process_market_update(self, updates: List[Dict[str, Any]]):
        """Process market updates from WebSocket."""
        for update in updates:
            event_type = update.get('event_type')

            if event_type not in ['book', 'price_change']:
                continue

            # Get market identifier
            market = update.get('market', '')
            asset_id = update.get('asset_id', '')

            # Find condition_id from market or asset_id
            condition_id = None
            for cid, info in self.sports_markets.items():
                if info.get('yes_token') == asset_id or info.get('no_token') == asset_id:
                    condition_id = cid
                    break

            if not condition_id:
                continue

            # Get market question
            question = self.market_questions.get(condition_id, 'Unknown market')

            # Extract price data
            if event_type == 'book':
                # Full order book
                bids = update.get('bids', [])
                asks = update.get('asks', [])

                if bids and asks:
                    best_bid = float(bids[0]['price']) if bids else 0.0
                    best_ask = float(asks[0]['price']) if asks else 0.0

                    # Update spike detector
                    spike = self.spike_detector.update_price(
                        market_id=condition_id,
                        market_question=question,
                        best_bid=best_bid,
                        best_ask=best_ask
                    )

                    if spike:
                        await self._handle_spike(spike)

            elif event_type == 'price_change':
                # Price change update
                # For spike detection, we need the full book, so skip price_change for now
                # In production, you might want to track these separately
                pass

    async def _handle_spike(self, spike: Spike):
        """Handle detected spike."""
        logger.info(f"Spike detected: {spike}")

        print(f"\n{'='*120}")
        print(f"🚨 SPIKE DETECTED!")
        print(f"{'='*120}")
        print(f"Market: {spike.market_question}")
        print(f"Direction: {spike.direction.upper()}")
        print(f"Price change: {spike.price_change_pct:+.2f}% ({spike.previous_price:.3f} → {spike.current_price:.3f})")
        print(f"Time window: {spike.time_window}s")
        print(f"Spike strength: {spike.spike_strength:.2f} std devs")

        # Find relevant news
        news_matches = self.news_monitor.match_to_market(spike.market_question, max_results=3)

        if news_matches:
            print(f"\nRelated news ({len(news_matches)} items):")
            for match in news_matches:
                item = match['news']
                print(f"  [{item.source}] {item.title}")
                print(f"    Relevance: {match['relevance_score']:.2f}")

        # LLM analysis (if available)
        if self.llm_analyzer and news_matches:
            print(f"\nRunning LLM analysis...")
            analysis = self.llm_analyzer.analyze_spike(
                market_question=spike.market_question,
                current_price=spike.current_price,
                previous_price=spike.previous_price,
                price_change_pct=spike.price_change_pct,
                news_items=[m['news'] for m in news_matches]
            )

            if analysis:
                print(f"\n📊 LLM Analysis:")
                print(f"  Justified: {'YES' if analysis.justified else 'NO'}")
                print(f"  Confidence: {analysis.confidence}%")
                print(f"  Recommendation: {analysis.recommendation.upper()}")
                print(f"  Reasoning: {analysis.reasoning}")

                if analysis.near_resolution:
                    print(f"  ⏰ Near resolution: {analysis.estimated_time_to_resolution}")

                should_trade = self.llm_analyzer.should_trade(analysis)
                print(f"\n  {'✅ WOULD TRADE' if should_trade else '❌ SKIP'} (in live mode)")

        print(f"{'='*120}\n")


async def run_live_scanner(config_path: str = 'spike_momentum/config.yaml'):
    """Run the live market scanner."""
    import yaml
    from spike_momentum.llm_provider import LLMProvider

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize Polymarket client (read-only, no trading)
    try:
        client = PolymarketClient()
    except ValueError:
        # No credentials, that's okay for scanning
        print("⚠️  No Polymarket credentials found - using public API only")
        print("   Some features may be limited")
        # You might want to handle this differently
        return

    # Initialize components
    news_monitor = SportsNewsMonitor(config.get('news', {}))
    spike_detector = SpikeDetector(config.get('spike_detection', {}))

    # LLM analyzer (optional)
    llm_analyzer = None
    if config.get('llm', {}).get('enabled', True):
        try:
            llm_provider = LLMProvider(config['llm'])
            llm_analyzer = LLMAnalyzer(llm_provider, config['llm'])
        except Exception as e:
            logger.warning(f"LLM analyzer disabled: {e}")

    # Initialize scanner
    scanner = MarketScanner(
        client=client,
        spike_detector=spike_detector,
        news_monitor=news_monitor,
        llm_analyzer=llm_analyzer,
        config=config
    )

    # Fetch initial news
    print("Fetching initial news...")
    news_items = news_monitor.fetch_news(max_items=20)
    print(f"✓ Loaded {len(news_items)} news items\n")

    # Start monitoring
    await scanner.monitor_markets()


if __name__ == '__main__':
    asyncio.run(run_live_scanner())
