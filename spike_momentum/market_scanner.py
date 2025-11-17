"""
Market Scanner for Spike Momentum Bot - ENHANCED VERSION

Features:
1. Proper pagination handling (fetches ALL available markets)
2. sportsMarketType-based filtering (moneyline markets only)
3. Keyword-based fallback with word boundary matching
4. Filters out spread/over-under/total markets
"""

import asyncio
import json
import re
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime

from poly_data.polymarket_client import PolymarketClient
from spike_momentum.spike_detector import SpikeDetector, Spike
from spike_momentum.news_monitor import SportsNewsMonitor
from spike_momentum.llm_analyzer import LLMAnalyzer
from spike_momentum.post_res_arb import PostResolutionArbitrage
from poly_utils.logging_utils import get_logger

logger = get_logger('spike_momentum.scanner')


class MarketScanner:
    """Scans Polymarket for sports markets and monitors for spikes."""

    # Sports-specific keywords with word boundaries
    SPORTS_KEYWORDS = [
        # Leagues
        r'\bnfl\b', r'\bnba\b', r'\bmlb\b', r'\bnhl\b', r'\bmls\b',
        r'\bncaa\b', r'\bncaaf\b', r'\bncaab\b',
        r'premier league', r'champions league', r'\buefa\b', r'world cup', r'la liga',
        # Sports (word boundaries to avoid false matches)
        r'\bfootball\b', r'\bbasketball\b', r'\bbaseball\b', r'\bhockey\b', r'\bsoccer\b',
        # Events
        r'\bplayoff', r'super bowl', r'world series', r'stanley cup', r'\bfinals\b',
        r'championship', r'tournament', r'bowl game',
        # NFL Teams
        r'\bchiefs\b', r'\bbills\b', r'\bcowboys\b', r'\bpatriots\b',
        r'\b49ers\b', r'\beagles\b', r'\bpackers\b',
        r'\bravens\b', r'\bdolphins\b', r'\bbengals\b', r'\bbrowns\b',
        r'\bsteelers\b', r'\bcolts\b', r'\btexans\b',
        # NBA Teams
        r'\blakers\b', r'\bwarriors\b', r'\bceltics\b', r'\bheat\b',
        r'\bbucks\b', r'\bnuggets\b', r'\bsuns\b',
        r'\bmavericks\b', r'\bclippers\b', r'\bknicks\b',
        r'\bbulls\b', r'\bjazz\b', r'\bgrizzlies\b', r'\bhawks\b',
        # Soccer Teams
        r'\bmanchester\b', r'\bliverpool\b', r'\bbarcelona\b', r'real madrid',
        r'\bchelsea\b', r'\barsenal\b', r'\bbayern\b', r'\bpsg\b',
        # Player names (high-profile only)
        r'\bmahomes\b', r'\blebron\b', r'\bcurry\b', r'\bmessi\b', r'\bronaldo\b',
    ]

    # Category-based filtering (most reliable)
    SPORTS_CATEGORIES = ['sports', 'football', 'basketball', 'baseball', 'soccer', 'hockey']

    # Tags that indicate sports
    SPORTS_TAGS = ['sports', 'nfl', 'nba', 'mlb', 'nhl', 'soccer', 'football', 'basketball']

    def __init__(
        self,
        client: PolymarketClient,
        spike_detector: SpikeDetector,
        news_monitor: SportsNewsMonitor,
        llm_analyzer: Optional[LLMAnalyzer] = None,
        post_res_arb: Optional[PostResolutionArbitrage] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize market scanner.

        Args:
            client: Polymarket client
            spike_detector: Spike detector instance
            news_monitor: News monitor instance
            llm_analyzer: Optional LLM analyzer
            post_res_arb: Optional post-resolution arb detector
            config: Optional configuration
        """
        self.client = client
        self.spike_detector = spike_detector
        self.news_monitor = news_monitor
        self.llm_analyzer = llm_analyzer
        self.post_res_arb = post_res_arb
        self.config = config or {}

        # Market data
        self.sports_markets = {}  # market_id -> market_info
        self.market_questions = {}  # market_id -> question text

        # Compile regex patterns once
        self.keyword_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.SPORTS_KEYWORDS]

        strategies = []
        if spike_detector.enabled:
            strategies.append("spike_momentum")
        if post_res_arb and post_res_arb.enabled:
            strategies.append("post_res_arb")

        logger.info(f"Initialized MarketScanner with strategies: {strategies}")

    def fetch_sports_markets(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Fetch sports markets from Polymarket.

        Args:
            limit: Maximum markets to return (None = no limit, return all)

        Returns:
            List of market dictionaries
        """
        logger.info("Fetching markets from Polymarket...")

        try:
            # Try to fetch with tag filtering first (if API supports it)
            # This is much more efficient than fetching all markets
            sports_markets = self._fetch_markets_by_tags()

            if sports_markets:
                logger.info(f"Fetched {len(sports_markets)} markets using tag filtering")
            else:
                # Fallback: fetch all and filter manually
                logger.info("Tag filtering not available, fetching all markets...")
                sports_markets = self._fetch_all_markets_and_filter(limit)

            # Apply limit if specified
            markets_to_store = sports_markets if limit is None else sports_markets[:limit]

            # Store market info
            for market in markets_to_store:
                condition_id = market.get('condition_id', '')
                question = market.get('question', '')

                # Get tokens for this market
                tokens = market.get('tokens', [])
                if len(tokens) >= 2:
                    yes_token = tokens[0].get('token_id', '')
                    no_token = tokens[1].get('token_id', '')

                    # Extract game metadata from events (real-time game status!)
                    events = market.get('events', [])
                    game_metadata = {}
                    if events and len(events) > 0:
                        event = events[0]
                        game_metadata = {
                            'live': event.get('live', False),
                            'ended': event.get('ended', False),
                            'score': event.get('score', ''),
                            'period': event.get('period', ''),
                            'elapsed': event.get('elapsed', ''),
                            'finishedTimestamp': event.get('finishedTimestamp', ''),
                            'startTime': event.get('startTime', ''),
                        }

                        # Log game metadata samples for data analysis
                        # Log if: live game, ended game, or has score/period data
                        if event.get('live') or event.get('ended') or event.get('score') or event.get('period'):
                            logger.info(f"🎮 GAME METADATA DETECTED [{condition_id[:8]}...]")
                            logger.info(f"   Market: {question[:80]}")
                            logger.info(f"   Event data: {json.dumps(event, indent=2)}")

                    self.sports_markets[condition_id] = {
                        'condition_id': condition_id,
                        'question': question,
                        'yes_token': yes_token,
                        'no_token': no_token,
                        'market_slug': market.get('market_slug', ''),
                        'category': market.get('category', ''),
                        'end_date': market.get('end_date_iso', ''),
                        'game_metadata': game_metadata,  # Real-time game status!
                    }

                    self.market_questions[condition_id] = question

            logger.info(f"Stored {len(self.sports_markets)} sports markets")
            return markets_to_store

        except Exception as e:
            logger.error(f"Error fetching sports markets: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _fetch_markets_by_tags(self) -> List[Dict[str, Any]]:
        """Try to fetch markets using tag/category filtering."""
        sports_markets = []

        # Try different tags
        for tag in self.SPORTS_TAGS:
            try:
                # Note: This might not be supported by current API version
                # Check py-clob-client documentation
                response = self.client.client.get_markets(tag=tag)
                if response and isinstance(response, list):
                    sports_markets.extend(response)
                    logger.info(f"Found {len(response)} markets with tag '{tag}'")
            except TypeError:
                # API doesn't support tag parameter, return empty list
                return []
            except Exception as e:
                logger.debug(f"Error fetching by tag '{tag}': {e}")
                continue

        return sports_markets

    def _fetch_all_markets_and_filter(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fallback: fetch all markets and filter manually with proper pagination."""
        cursor = ""
        all_markets = []
        iterations = 0
        max_iterations = 50  # Safety limit to prevent infinite loops

        # Fetch ALL markets using pagination (like update_markets.py does)
        while iterations < max_iterations:
            try:
                response = self.client.client.get_sampling_markets(next_cursor=cursor)
                markets_data = response.get('data', [])

                if not markets_data:
                    break

                all_markets.extend(markets_data)

                cursor = response.get('next_cursor')
                if cursor is None:
                    break

                iterations += 1

            except Exception as e:
                # API sometimes fails with bad cursor after exhausting markets
                # This is expected behavior at the end of pagination
                logger.debug(f"Pagination ended (fetched {len(all_markets)} markets): {e}")
                break

        logger.info(f"Fetched {len(all_markets)} total markets")

        # Filter for sports markets (moneyline only)
        sports_markets = []
        moneyline_count = 0
        filtered_count = 0

        for market in all_markets:
            # Check for sportsMarketType at market level (not in events!)
            sports_type = market.get('sportsMarketType')
            if sports_type:
                if sports_type.lower() == 'moneyline':
                    moneyline_count += 1
                else:
                    filtered_count += 1

            if self._is_sports_market(market):
                sports_markets.append(market)

        logger.info(f"Found {len(sports_markets)} sports markets after filtering")
        if moneyline_count > 0:
            logger.info(f"  ├─ {moneyline_count} moneyline markets (via sportsMarketType)")
        if filtered_count > 0:
            logger.info(f"  ├─ {filtered_count} non-moneyline sports markets (filtered out)")
        if len(sports_markets) > moneyline_count:
            logger.info(f"  └─ {len(sports_markets) - moneyline_count} markets (via keyword matching)")

        return sports_markets

    def _is_sports_market(self, market: Dict[str, Any]) -> bool:
        """
        Check if market is sports-related using sportsMarketType (preferred) or keywords.

        Only accepts moneyline markets for simplicity.
        """
        # PRIORITY 1: Check for sportsMarketType field at market level (most reliable!)
        # This field indicates real sports markets and their type
        sports_market_type = market.get('sportsMarketType')

        if sports_market_type:
            # Only trade moneyline markets (most straightforward)
            if sports_market_type.lower() == 'moneyline':
                logger.debug(f"Found moneyline sports market: {market.get('question', '')[:50]}...")
                return True
            else:
                # Skip spreads, over/under, parlays, etc.
                logger.debug(f"Skipping non-moneyline sports market ({sports_market_type}): {market.get('question', '')[:50]}...")
                return False

        # FALLBACK: Use keyword-based detection if sportsMarketType not available
        question = market.get('question', '').lower()
        category = market.get('category', '').lower()
        tags = market.get('tags', [])

        # Check category (most reliable fallback)
        if category in self.SPORTS_CATEGORIES:
            # Additional check: avoid spread/over-under markets in keywords
            if any(keyword in question for keyword in ['spread', 'over ', 'under ', 'o/u', 'total points']):
                logger.debug(f"Skipping spread/total market: {market.get('question', '')[:50]}...")
                return False
            return True

        # Check tags
        if tags:
            for tag in tags:
                if tag.lower() in self.SPORTS_TAGS:
                    if any(keyword in question for keyword in ['spread', 'over ', 'under ', 'o/u', 'total points']):
                        return False
                    return True

        # Check question with word boundary patterns
        for pattern in self.keyword_patterns:
            if pattern.search(question):
                if any(keyword in question for keyword in ['spread', 'over ', 'under ', 'o/u', 'total points']):
                    return False
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

        # Fetch all sports markets (no limit - monitor everything we find)
        sports_markets = self.fetch_sports_markets(limit=None)

        if not sports_markets:
            logger.error("No sports markets found to monitor")
            return

        # Get token IDs to subscribe to
        token_ids = self.get_token_ids()

        if not token_ids:
            logger.error("No token IDs to monitor")
            return

        print(f"\n✓ Monitoring {len(self.sports_markets)} sports markets ({len(token_ids)} tokens)")
        print("\nSample markets:")
        for i, (cid, info) in enumerate(list(self.sports_markets.items())[:5]):
            print(f"  {i+1}. {info['question'][:80]}")
        print()

        # Connect to WebSocket and process updates
        uri = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

        try:
            import websockets

            async with websockets.connect(uri, ping_interval=5, ping_timeout=None) as websocket:
                # Subscribe to markets
                message = {"assets_ids": token_ids}
                await websocket.send(json.dumps(message))

                logger.info(f"Subscribed to {len(token_ids)} token IDs")
                print(f"✓ Connected to WebSocket\n")
                print("Waiting for price spikes...\n")

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
                    mid_price = (best_bid + best_ask) / 2

                    # Check for spike momentum opportunity
                    if self.spike_detector.enabled:
                        spike = self.spike_detector.update_price(
                            market_id=condition_id,
                            market_question=question,
                            best_bid=best_bid,
                            best_ask=best_ask
                        )

                        if spike:
                            await self._handle_spike(spike)

                    # Check for post-resolution arb opportunity
                    if self.post_res_arb and self.post_res_arb.enabled:
                        market_info = self.sports_markets.get(condition_id, {})
                        arb_opp = await self.post_res_arb.check_market_for_arb(
                            market_id=condition_id,
                            market_question=question,
                            current_price=mid_price,
                            market_info=market_info
                        )

                        if arb_opp:
                            await self._handle_post_res_arb(arb_opp)

            elif event_type == 'price_change':
                # Price change update - could implement incremental updates here
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
        else:
            print(f"\nNo related news found")

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

    async def _handle_post_res_arb(self, opportunity: Dict[str, Any]):
        """Handle post-resolution arb opportunity."""
        logger.info(f"Post-res arb opportunity: {opportunity}")

        # Execute arb (dry-run for now)
        dry_run = self.config.get('operation', {}).get('dry_run', True)
        result = await self.post_res_arb.execute_arb(opportunity, dry_run=dry_run)

        if result.get('success'):
            logger.info(f"Arb executed successfully: {result}")
        else:
            logger.warning(f"Arb execution failed: {result.get('error')}")


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
    except ValueError as e:
        print(f"⚠️  No Polymarket credentials: {e}")
        print("   Some features may be limited")
        return

    # Initialize components
    news_monitor = SportsNewsMonitor(config.get('news', {}))
    spike_detector = SpikeDetector(config.get('spike_detection', {}))

    # LLM provider (shared across strategies)
    llm_provider = None
    if config.get('llm', {}).get('enabled', True):
        try:
            llm_provider = LLMProvider(config['llm'])
            print(f"✓ LLM provider initialized ({config['llm']['provider']})\n")
        except Exception as e:
            logger.warning(f"LLM provider disabled: {e}")
            print(f"⚠️  LLM provider disabled: {e}\n")

    # LLM analyzer (for spike momentum)
    llm_analyzer = None
    if llm_provider:
        llm_analyzer = LLMAnalyzer(llm_provider, config['llm'])

    # Post-resolution arb (separate strategy)
    post_res_arb = None
    if config.get('post_resolution_arb', {}).get('enabled', False):
        post_res_arb = PostResolutionArbitrage(
            client=client,
            news_monitor=news_monitor,
            llm_provider=llm_provider,
            config=config
        )
        print(f"✓ Post-resolution arb initialized\n")

    # Display capital allocation
    capital_config = config.get('capital', {})
    if capital_config:
        total = capital_config.get('total_capital', 0)
        spike_pct = capital_config.get('spike_momentum', 0) * 100
        arb_pct = capital_config.get('post_resolution_arb', 0) * 100
        print(f"Capital Allocation (${total:.0f} total):")
        print(f"  Spike Momentum: {spike_pct:.0f}% (${total * spike_pct/100:.0f})")
        print(f"  Post-Res Arb: {arb_pct:.0f}% (${total * arb_pct/100:.0f})\n")

    # Initialize scanner
    scanner = MarketScanner(
        client=client,
        spike_detector=spike_detector,
        news_monitor=news_monitor,
        llm_analyzer=llm_analyzer,
        post_res_arb=post_res_arb,
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
