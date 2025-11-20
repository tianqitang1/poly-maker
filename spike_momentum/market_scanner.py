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
import aiohttp

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
        self._live_poll_offset = 0  # round-robin cursor for normal priority polling batches
        self._started_poll_offset = 0 # round-robin cursor for started games polling batches

        # Compile regex patterns once
        self.keyword_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.SPORTS_KEYWORDS]
        
        self._should_reconnect_ws = False

        strategies = []
        if spike_detector.enabled:
            strategies.append("spike_momentum")
        if post_res_arb and post_res_arb.enabled:
            strategies.append("post_res_arb")

        logger.info(f"Initialized MarketScanner with strategies: {strategies}")

    async def fetch_sports_markets(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Fetch sports markets from Polymarket and update internal state.

        Args:
            limit: Maximum markets to return (None = no limit, return all)

        Returns:
            List of market dictionaries
        """
        new_sports_markets, new_market_questions, markets_to_store = await self._fetch_sports_markets_data(limit)
        
        # Update state
        self.sports_markets = new_sports_markets
        self.market_questions = new_market_questions
        
        logger.info(f"Stored {len(self.sports_markets)} sports markets")
        return markets_to_store

    async def _fetch_markets_from_gamma(self) -> List[Dict[str, Any]]:
        """
        Fetch markets directly from Gamma API using sports tags.
        This is more reliable for discovery than CLOB pagination.
        """
        logger.info("Fetching sports markets from Gamma API...")
        
        sports_tags = ['nba', 'nfl', 'mlb', 'nhl', 'soccer', 'football', 'basketball', 'sports']
        found_markets = {} # Dedup by condition_id
        
        async with aiohttp.ClientSession() as session:
            for tag in sports_tags:
                try:
                    # Fetch events for this tag
                    url = f"https://gamma-api.polymarket.com/events?limit=50&active=true&closed=false&tag_slug={tag}"
                    async with session.get(url) as response:
                        if response.status != 200:
                            continue
                            
                        events = await response.json()
                        if not isinstance(events, list):
                            continue
                            
                        for event in events:
                            markets = event.get('markets', [])
                            for market in markets:
                                # Filter for moneyline sports markets
                                # Gamma markets have 'sportsMarketType'
                                if market.get('sportsMarketType') != 'moneyline':
                                    continue
                                    
                                # Extract needed fields to match CLOB format
                                condition_id = market.get('conditionId')
                                if not condition_id or condition_id in found_markets:
                                    continue
                                    
                                # Construct market dict
                                found_markets[condition_id] = {
                                    'condition_id': condition_id,
                                    'question': market.get('question'),
                                    'market_slug': market.get('slug'),
                                    'category': 'Sports', # Assumed from tag
                                    'end_date_iso': market.get('endDate'),
                                    'gameStartTime': market.get('gameStartTime'),
                                    'rewards': {'min_size': 0, 'max_spread': 0, 'rates': []}, # Defaults
                                    'tokens': json.loads(market.get('outcomes', '[]')), # Outcomes only, tokens needed?
                                    # Gamma doesn't give token IDs directly in the simple list sometimes
                                    # We need clobTokenIds
                                    'clobTokenIds': json.loads(market.get('clobTokenIds', '[]')),
                                    'events': [event], # Attach full event for metadata extraction
                                    'sportsMarketType': 'moneyline'
                                }
                                
                except Exception as e:
                    logger.debug(f"Error fetching Gamma tag {tag}: {e}")
                    
        # Convert to list and normalize
        results = []
        for m in found_markets.values():
            # Normalize token IDs from clobTokenIds
            clob_ids = m.pop('clobTokenIds', [])
            outcomes = m.pop('tokens', [])
            
            tokens = []
            if len(clob_ids) >= 2 and len(outcomes) >= 2:
                tokens = [
                    {'token_id': clob_ids[0], 'outcome': outcomes[0]},
                    {'token_id': clob_ids[1], 'outcome': outcomes[1]}
                ]
            m['tokens'] = tokens
            results.append(m)
            
        logger.info(f"Gamma fetch found {len(results)} unique sports markets")
        return results

    async def _fetch_sports_markets_data(self, limit: Optional[int] = None) -> tuple[Dict[str, Any], Dict[str, str], List[Dict[str, Any]]]:
        """
        Internal method to fetch sports markets without updating state.
        Returns (sports_markets_dict, market_questions_dict, list_of_markets).
        """
        logger.info("Fetching markets from Polymarket...")

        new_sports_markets = {}
        new_market_questions = {}
        markets_to_store = []

        try:
            # PRIORITY 1: Fetch from Gamma API (New, reliable method)
            try:
                sports_markets_list = await self._fetch_markets_from_gamma()
            except Exception as e:
                logger.error(f"Gamma fetch failed: {e}")
                sports_markets_list = []

            if not sports_markets_list:
                # PRIORITY 2: Try tag filtering on CLOB (Legacy)
                sports_markets_list = self._fetch_markets_by_tags()

                if sports_markets_list:
                    logger.info(f"Fetched {len(sports_markets_list)} markets using tag filtering")
                else:
                    # PRIORITY 3: Fallback to all markets (Slow, error-prone)
                    logger.info("Tag filtering not available, fetching all markets...")
                    sports_markets_list = self._fetch_all_markets_and_filter(limit)

            # Apply limit if specified
            markets_to_store = sports_markets_list if limit is None else sports_markets_list[:limit]

            # Store market info
            for market in markets_to_store:
                condition_id = market.get('condition_id') or market.get('conditionId')
                if not condition_id: 
                    continue
                    
                question = market.get('question', '')

                # Get tokens for this market
                tokens = market.get('tokens', [])
                yes_token = ""
                no_token = ""
                
                if len(tokens) >= 2:
                    yes_token = tokens[0].get('token_id', '')
                    no_token = tokens[1].get('token_id', '')
                elif market.get('clobTokenIds'):
                     # Handle raw format from Gamma if tokens struct wasn't built perfectly
                     cids = market.get('clobTokenIds')
                     if isinstance(cids, str): cids = json.loads(cids)
                     if len(cids) >= 2:
                         yes_token = cids[0]
                         no_token = cids[1]

                if yes_token and no_token:
                    # Slug used for live-game polling endpoint; fall back through common keys
                    market_slug = (
                        market.get('market_slug')
                        or market.get('slug')
                        or market.get('ticker')
                        or ''
                    )

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

                    new_sports_markets[condition_id] = {
                        'condition_id': condition_id,
                        'question': question,
                        'yes_token': yes_token,
                        'no_token': no_token,
                        'market_slug': market_slug,
                        'category': market.get('category', ''),
                        'end_date': market.get('end_date_iso') or market.get('endDate'),
                        'game_start_time': market.get('gameStartTime') or game_metadata.get('startTime'),
                        'game_metadata': game_metadata,  # Real-time game status!
                    }

                    new_market_questions[condition_id] = question

            return new_sports_markets, new_market_questions, markets_to_store

        except Exception as e:
            logger.error(f"Error fetching sports markets: {e}")
            import traceback
            traceback.print_exc()
            return {}, {}, []

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
        page = 0
        max_iterations = 50  # Safety limit to prevent infinite loops

        # Fetch ALL markets using pagination (like update_markets.py does)
        while page < max_iterations:
            try:
                response = self.client.client.get_sampling_markets(next_cursor=cursor)
                markets_data = response.get('data', [])

                if not markets_data:
                    break

                all_markets.extend(markets_data)
                page += 1

                # Progress logging (like near_sure does)
                logger.info(f"Fetched page {page}: {len(markets_data)} markets (Total: {len(all_markets)})")

                cursor = response.get('next_cursor')
                if cursor is None:
                    logger.info(f"Pagination complete: {page} pages fetched")
                    break

            except Exception as e:
                # API sometimes fails with bad cursor after exhausting markets
                # This is expected behavior at the end of pagination
                logger.info(f"Pagination ended after {page} pages (fetched {len(all_markets)} markets): {e}")
                break

        logger.info(f"Fetched {len(all_markets)} total markets, now filtering for sports...")

        # Filter for sports markets (moneyline only)
        sports_markets = []
        moneyline_count = 0
        filtered_count = 0

        # Progress tracking for filtering (show every 500 markets)
        processed = 0
        for market in all_markets:
            processed += 1
            if processed % 500 == 0:
                logger.info(f"Filtering progress: {processed}/{len(all_markets)} markets checked...")

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
            # Skip debug logging to speed up filtering (called ~3000 times)
            return sports_market_type.lower() == 'moneyline'

        # FALLBACK: Use keyword-based detection if sportsMarketType not available
        question = market.get('question', '').lower()
        category = market.get('category', '').lower()
        tags = market.get('tags', [])

        # Early check: avoid spread/over-under markets
        spread_keywords = ['spread', 'over ', 'under ', 'o/u', 'total points']
        is_spread_market = any(keyword in question for keyword in spread_keywords)

        # Check category (most reliable fallback)
        if category in self.SPORTS_CATEGORIES:
            return not is_spread_market

        # Check tags
        if tags and any(tag.lower() in self.SPORTS_TAGS for tag in tags):
            return not is_spread_market

        # Check question with word boundary patterns
        if any(pattern.search(question) for pattern in self.keyword_patterns):
            return not is_spread_market

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

    async def _fetch_live_game_status(self, market_slug: str) -> Optional[Dict[str, Any]]:
        """
        Fetch live game status from Gamma API for a specific market.

        Args:
            market_slug: The market slug identifier

        Returns:
            Game metadata dict with live, score, period, etc. or None if error
        """
        try:
            url = f"https://gamma-api.polymarket.com/markets/slug/{market_slug}"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status != 200:
                        logger.debug(f"Failed to fetch {market_slug}: HTTP {response.status}")
                        return None

                    data = await response.json()

                    # Extract events data (same structure as manual curl)
                    events = data.get('events', [])
                    if not events or len(events) == 0:
                        logger.debug(f"No events array for {market_slug}")
                        # Return empty metadata to indicate "not a live game market"
                        return {
                            'live': False,
                            'ended': False,
                            'score': '',
                            'period': '',
                            'elapsed': '',
                            'startTime': '',
                            'finishedTimestamp': '',
                        }

                    event = events[0]
                    game_metadata = {
                        'live': event.get('live', False),
                        'ended': event.get('ended', False),
                        'score': event.get('score', ''),
                        'period': event.get('period', ''),
                        'elapsed': event.get('elapsed', ''),
                        'startTime': event.get('startTime', ''),
                        'finishedTimestamp': event.get('finishedTimestamp', ''),
                    }

                    # Log if we found a live or ended game
                    if game_metadata.get('live') or game_metadata.get('ended'):
                        logger.info(f"Found {'LIVE' if game_metadata.get('live') else 'ENDED'} game: {market_slug}")

                    return game_metadata

        except asyncio.TimeoutError:
            logger.debug(f"Timeout fetching live status for {market_slug}")
            return None
        except Exception as e:
            logger.debug(f"Error fetching live status for {market_slug}: {e}")
            return None

    async def _update_live_games_loop(self):
        """
        Background task that periodically updates live game status for all markets.
        Runs every 30 seconds to check for live games.
        """
        logger.info("Starting live game status updater...")

        while True:
            try:
                # Wait 30 seconds between updates
                await asyncio.sleep(30)

                # Find markets that might be live (have game_start_time in the past)
                now = datetime.utcnow()

                # Build prioritized list of candidates
                actually_live = []    # Priority 0: Metadata says LIVE (check every loop!)
                recent_starts = []    # Priority 1: Started < 6h ago (likely live/just finished)
                old_starts = []       # Priority 2: Started > 6h ago (check for late resolution)
                normal_priority = []  # Priority 3: Future games (check occasionally, rotated)

                for condition_id, market_info in self.sports_markets.items():
                    # Skip if already ended AND finished > 12 hours ago
                    # We want to keep checking recently ended games for arb opportunities
                    game_metadata = market_info.get('game_metadata', {})
                    if game_metadata.get('ended'):
                        should_skip = False
                        finished_ts = game_metadata.get('finishedTimestamp')
                        
                        if finished_ts:
                            try:
                                from dateutil import parser
                                finished_time = parser.parse(finished_ts).replace(tzinfo=None)
                                hours_since_finish = (now - finished_time).total_seconds() / 3600
                                
                                # specific check: if > 12 hours, we're done with it
                                if hours_since_finish > 12:
                                    should_skip = True
                            except Exception:
                                # If we can't parse timestamp, don't skip (safety)
                                pass
                        
                        if should_skip:
                            continue

                    market_slug = market_info.get('market_slug')
                    if not market_slug:
                        continue
                        
                    # Check if game is actually live per metadata
                    if game_metadata.get('live'):
                        actually_live.append((condition_id, market_slug))
                        continue

                    # Check if game has started (using gameStartTime)
                    game_start_time_str = market_info.get('game_start_time', '')
                    time_since_start = 0
                    is_started = False

                    if game_start_time_str:
                        try:
                            from dateutil import parser
                            game_start_time = parser.parse(game_start_time_str).replace(tzinfo=None)
                            time_since_start = (now - game_start_time).total_seconds()

                            # Game started in the past
                            if time_since_start > 0:
                                is_started = True
                        except Exception:
                            pass

                    if is_started:
                        # Prioritize recent starts (within 6 hours)
                        if time_since_start < 6 * 3600:
                            recent_starts.append((condition_id, market_slug))
                        else:
                            old_starts.append((condition_id, market_slug))
                    else:
                        normal_priority.append((condition_id, market_slug))

                # Combine: 
                # 1. All actually_live games (limit to batch_size)
                # 2. recent_starts (High priority!)
                # 3. old_starts (Round Robin)
                # 4. normal_priority (Round Robin)
                
                batch_size = 80  # Increased to 80 (~2.6 req/s) for better coverage
                candidates_to_check = []

                # 1. Actually Live (Take ALL up to batch limit)
                candidates_to_check.extend(actually_live[:batch_size])
                
                remaining_slots = batch_size - len(candidates_to_check)

                # 2. Recent Starts (Take ALL up to remaining slots)
                # These are critical - likely the "ongoing games" user is missing
                if remaining_slots > 0 and recent_starts:
                    take = recent_starts[:remaining_slots]
                    candidates_to_check.extend(take)
                    remaining_slots -= len(take)

                # 3. Old Starts (Round Robin)
                if remaining_slots > 0 and old_starts:
                    pool = old_starts
                    count = len(pool)
                    offset = self._started_poll_offset
                    
                    take_count = min(remaining_slots, count)
                    start = offset % count
                    batch = pool[start : start + take_count]
                    
                    # Wrap around
                    if len(batch) < take_count:
                        wrap_needed = take_count - len(batch)
                        batch.extend(pool[:wrap_needed])
                        
                    candidates_to_check.extend(batch)
                    self._started_poll_offset = (start + take_count) % count
                    remaining_slots -= len(batch)

                # 4. Normal Priority (Round Robin)
                if remaining_slots > 0 and normal_priority:
                    pool = normal_priority
                    count = len(pool)
                    offset = self._live_poll_offset
                    
                    take_count = min(remaining_slots, count)
                    start = offset % count
                    batch = pool[start : start + take_count]
                    
                    # Wrap around
                    if len(batch) < take_count:
                        wrap_needed = take_count - len(batch)
                        batch.extend(pool[:wrap_needed])
                        
                    candidates_to_check.extend(batch)
                    self._live_poll_offset = (start + take_count) % count

                logger.info(
                    f"Poll Stats: Live={len(actually_live)}, "
                    f"Recent={len(recent_starts)} (<6h), "
                    f"Old={len(old_starts)}, "
                    f"Future={len(normal_priority)}"
                )
                logger.info(f"Checking {len(candidates_to_check)} markets for status...")

                live_count = 0
                ended_count = 0
                checked_count = 0

                for condition_id, market_slug in candidates_to_check:
                    checked_count += 1
                    game_metadata = await self._fetch_live_game_status(market_slug)

                    if game_metadata:
                        # Update market metadata
                        self.sports_markets[condition_id]['game_metadata'] = game_metadata

                        if game_metadata.get('live'):
                            live_count += 1
                            logger.debug(
                                f"Live game: {self.sports_markets[condition_id]['question'][:50]} - "
                                f"Score: {game_metadata.get('score', 'N/A')}, "
                                f"Period: {game_metadata.get('period', 'N/A')}"
                            )

                        if game_metadata.get('ended'):
                            ended_count += 1
                            logger.info(
                                f"🏁 Game ended: {self.sports_markets[condition_id]['question'][:50]} - "
                                f"Final score: {game_metadata.get('score', 'N/A')}"
                            )

                        # Check for post-resolution arb on live or ended games
                        if (game_metadata.get('live') or game_metadata.get('ended')) and self.post_res_arb:
                            if self.post_res_arb.enabled:
                                # Fetch current order book to get price
                                try:
                                    market_info = self.sports_markets[condition_id]
                                    yes_token = market_info.get('yes_token')
                                    no_token = market_info.get('no_token')

                                    if yes_token and no_token:
                                        book = {}
                                        try:
                                            if hasattr(self.client, 'get_order_book_dict'):
                                                book = self.client.get_order_book_dict(yes_token)
                                            else:
                                                book = self.client.get_order_book(yes_token)
                                        except Exception as e:
                                            logger.warning(f"Order book fetch failed for {yes_token}: {e}")

                                        bids = book.get('bids', []) if isinstance(book, dict) else []
                                        asks = book.get('asks', []) if isinstance(book, dict) else []

                                        # Legacy tuple/DataFrame support
                                        if not bids and not asks and isinstance(book, tuple) and len(book) == 2:
                                            bids_df, asks_df = book
                                            try:
                                                bids = bids_df.to_dict('records') if not bids_df.empty else []
                                                asks = asks_df.to_dict('records') if not asks_df.empty else []
                                            except Exception:
                                                bids, asks = [], []

                                        if bids and asks:
                                            best_bid = float(bids[0]['price']) if bids else None
                                            best_ask = float(asks[0]['price']) if asks else None

                                            if best_bid is not None and best_ask is not None:
                                                mid_price = (best_bid + best_ask) / 2
                                                
                                                # Update spread dynamically
                                                current_spread = best_ask - best_bid
                                                market_info['spread'] = current_spread

                                                arb_opp = await self.post_res_arb.check_market_for_arb(
                                                    market_id=condition_id,
                                                    market_question=market_info['question'],
                                                    current_price=mid_price,
                                                    market_info=market_info
                                                )

                                                if arb_opp:
                                                    await self._handle_post_res_arb(arb_opp)
                                except Exception as e:
                                    logger.error(f"Error checking post-res arb for {condition_id[:8]}...: {e}")

                logger.info(f"Live game polling: checked {checked_count} markets, found {live_count} live, {ended_count} ended")

                if live_count == 0 and ended_count == 0 and checked_count > 0:
                    logger.debug(f"No live or ended games found in {checked_count} markets checked")

            except Exception as e:
                logger.error(f"Error in live game updater: {e}")
                import traceback
                traceback.print_exc()

    async def _refresh_markets_loop(self):
        """
        Background task to periodically refresh the full market list.
        Runs every 60 minutes to pick up new games and remove old ones.
        """
        logger.info("Starting market refresh loop (60min interval)...")
        
        while True:
            try:
                # Wait 60 minutes
                await asyncio.sleep(3600)
                
                logger.info("🔄 REFRESHING MARKET LIST...")
                
                # Fetch new data (without updating state yet)
                new_sports_markets, new_market_questions, _ = await self._fetch_sports_markets_data(limit=None)
                
                if not new_sports_markets:
                    logger.warning("Market refresh failed or returned no markets. Keeping old state.")
                    continue
                    
                # MERGE STATE: Preserve game_metadata from old state for continuity
                # This ensures we don't lose track of 'live' status during the swap
                preserved_count = 0
                for condition_id, new_info in new_sports_markets.items():
                    if condition_id in self.sports_markets:
                        old_info = self.sports_markets[condition_id]
                        # If old info has live game metadata, copy it over
                        old_meta = old_info.get('game_metadata', {})
                        if old_meta and (old_meta.get('live') or old_meta.get('ended')):
                            new_info['game_metadata'] = old_meta
                            preserved_count += 1
                            
                logger.info(f"Preserved game metadata for {preserved_count} markets")
                
                # ATOMIC SWAP
                self.sports_markets = new_sports_markets
                self.market_questions = new_market_questions
                
                logger.info(f"✓ Market list refreshed: {len(self.sports_markets)} active markets")
                
                # Trigger WebSocket reconnect to subscribe to new tokens
                self._should_reconnect_ws = True
                
            except Exception as e:
                logger.error(f"Error in market refresh loop: {e}")
                import traceback
                traceback.print_exc()

    async def monitor_markets(self):
        """
        Monitor markets via WebSocket and detect spikes.

        This is the main monitoring loop with automatic reconnection.
        """
        logger.info("Starting market monitoring...")

        # Fetch all sports markets (no limit - monitor everything we find)
        # Initial load
        await self.fetch_sports_markets(limit=None)

        if not self.sports_markets:
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

        # Start background task for live game updates
        live_game_task = asyncio.create_task(self._update_live_games_loop())
        logger.info("Started live game status monitoring")
        
        # Start background task for market list refresh
        refresh_task = asyncio.create_task(self._refresh_markets_loop())
        logger.info("Started background market list refresh loop")

        # Reconnection loop with exponential backoff
        reconnect_delay = 1  # Start with 1 second
        max_reconnect_delay = 60  # Max 60 seconds

        while True:
            try:
                # Reset flag before connecting
                self._should_reconnect_ws = False
                
                # Refresh token IDs in case market list changed
                current_token_ids = self.get_token_ids()
                
                await self._connect_and_monitor(current_token_ids)
                # If we get here, connection closed gracefully - reset delay
                reconnect_delay = 1

            except Exception as e:
                logger.error(f"WebSocket connection error: {e}")
                logger.info(f"Reconnecting in {reconnect_delay} seconds...")
                await asyncio.sleep(reconnect_delay)

                # Exponential backoff
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

    async def _connect_and_monitor(self, token_ids: List[str]):
        """Connect to WebSocket and monitor until connection drops."""
        import websockets
        from websockets.exceptions import ConnectionClosed, WebSocketException

        uri = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

        async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as websocket:
            # Subscribe to markets
            message = {"assets_ids": token_ids}
            await websocket.send(json.dumps(message))

            logger.info(f"Subscribed to {len(token_ids)} token IDs")
            print(f"✓ Connected to WebSocket\n")
            print("Waiting for price spikes...\n")

            # Process incoming messages
            while True:
                # Check if we need to force reconnect (e.g. due to market refresh)
                if self._should_reconnect_ws:
                    logger.info("Forcing WebSocket reconnect due to market list refresh...")
                    return

                try:
                    message = await websocket.recv()
                    json_data = json.loads(message)

                    # Process market updates
                    if isinstance(json_data, dict):
                        await self._process_market_update([json_data])
                    elif isinstance(json_data, list):
                        await self._process_market_update(json_data)

                except (ConnectionClosed, WebSocketException) as e:
                    # WebSocket connection lost - raise to trigger reconnection
                    logger.warning(f"WebSocket connection closed: {e}")
                    raise

                except json.JSONDecodeError as e:
                    # Invalid JSON - log and continue
                    logger.warning(f"Invalid JSON received: {e}")
                    continue

                except Exception as e:
                    # Other errors - log and continue
                    logger.error(f"Error processing message: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

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

        # Display live game status if available
        market_info = self.sports_markets.get(spike.market_id)
        if market_info:
            game_metadata = market_info.get('game_metadata', {})
            if game_metadata and (game_metadata.get('live') or game_metadata.get('score')):
                print(f"\n🏟️  Live Game Status:")
                if game_metadata.get('live'):
                    print(f"  Status: 🔴 LIVE")
                elif game_metadata.get('ended'):
                    print(f"  Status: ✅ ENDED")
                else:
                    print(f"  Status: Pre-game")

                if game_metadata.get('score'):
                    print(f"  Score: {game_metadata['score']}")
                if game_metadata.get('period'):
                    print(f"  Period: {game_metadata['period']}")
                if game_metadata.get('elapsed'):
                    print(f"  Time: {game_metadata['elapsed']}")

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

        # LLM analysis (if available) - runs even without news if we have game data
        if self.llm_analyzer:
            # Only run LLM if we have game metadata OR news
            if game_metadata or news_matches:
                print(f"\nRunning LLM analysis...")
                analysis = self.llm_analyzer.analyze_spike(
                    market_question=spike.market_question,
                    current_price=spike.current_price,
                    previous_price=spike.previous_price,
                    price_change_pct=spike.price_change_pct,
                    game_metadata=game_metadata,
                    news_items=[m['news'] for m in news_matches] if news_matches else None
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
