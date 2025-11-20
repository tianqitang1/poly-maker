import logging
import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import aiohttp

from poly_data.polymarket_client import PolymarketClient
from poly_utils.logging_utils import get_logger

logger = get_logger('sports_arb.scanner')

class SportsMarketScanner:
    """
    Simplified market scanner focused specifically on Live and Recent Sports games.
    """
    
    SPORTS_TAGS = ['nba', 'nfl', 'mlb', 'nhl', 'soccer', 'football', 'basketball', 'sports']

    def __init__(self, client: PolymarketClient, config: Dict[str, Any]):
        self.client = client
        self.config = config
        self.markets: Dict[str, Dict[str, Any]] = {} # condition_id -> market_info
        
    async def fetch_active_sports_markets(self) -> List[Dict[str, Any]]:
        """
        Fetch active sports markets from Gamma API.
        Prioritizes fetching by tags to get all relevant sports.
        """
        logger.info("Fetching active sports markets...")
        found_markets = {}
        
        async with aiohttp.ClientSession() as session:
            for tag in self.SPORTS_TAGS:
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
                            self._process_event(event, found_markets)
                            
                except Exception as e:
                    logger.error(f"Error fetching tag {tag}: {e}")
        
        # Store and return
        self.markets = found_markets
        logger.info(f"Found {len(self.markets)} active sports markets")
        return list(self.markets.values())

    def _process_event(self, event: Dict[str, Any], markets_dict: Dict[str, Any]):
        """Extract moneyline markets from an event."""
        event_markets = event.get('markets', [])
        
        # Get event-level game data
        game_id = event.get('gameId')
        score = event.get('score')
        period = event.get('period')
        live = event.get('live', False)
        ended = event.get('ended', False)
        
        for market in event_markets:
            # We only want Moneyline (Winner) markets
            if market.get('sportsMarketType') != 'moneyline':
                continue
                
            condition_id = market.get('conditionId')
            if not condition_id:
                continue
                
            # Parse outcomes/tokens
            clob_ids = json.loads(market.get('clobTokenIds', '[]'))
            outcomes = json.loads(market.get('outcomes', '[]'))
            
            if len(clob_ids) != 2:
                continue
                
            market_info = {
                'condition_id': condition_id,
                'question': market.get('question'),
                'slug': market.get('slug'),
                'game_id': game_id,
                'live': live,
                'ended': ended,
                'score': score,
                'period': period,
                'start_time': market.get('gameStartTime'),
                'end_date': market.get('endDate'),
                'tokens': [
                    {'id': clob_ids[0], 'outcome': outcomes[0] if len(outcomes)>0 else 'Yes'},
                    {'id': clob_ids[1], 'outcome': outcomes[1] if len(outcomes)>1 else 'No'}
                ],
                'raw_event': event # Keep raw event for deep updates
            }
            
            markets_dict[condition_id] = market_info

    async def update_live_scores(self):
        """
        Polling loop to update scores for known live/recent markets.
        This provides the 'ground truth' for the Post-Resolution Arb strategy.
        """
        while True:
            try:
                # Identify priority markets (Live or Recently Started)
                targets = []
                now = datetime.utcnow()
                
                for cid, m in self.markets.items():
                    # Always check if flagged live/ended
                    if m['live'] or m['ended']:
                        targets.append(m)
                        continue
                        
                    # Check start time
                    start = m.get('start_time')
                    if start:
                        try:
                            from dateutil import parser
                            dt = parser.parse(start).replace(tzinfo=None)
                            diff = (now - dt).total_seconds()
                            # Check if started within last 12 hours
                            if 0 < diff < 12 * 3600:
                                targets.append(m)
                        except:
                            pass
                
                # Batch update (Gamma API doesn't have bulk endpoint for this, so we loop)
                # We limit concurrency
                if targets:
                    logger.info(f"Updating scores for {len(targets)} live/recent games...")
                    async with aiohttp.ClientSession() as session:
                        for market in targets:
                            slug = market['slug']
                            url = f"https://gamma-api.polymarket.com/markets/slug/{slug}"
                            try:
                                async with session.get(url) as resp:
                                    if resp.status == 200:
                                        data = await resp.json()
                                        events = data.get('events', [])
                                        if events:
                                            e = events[0]
                                            # Update state in place
                                            market['live'] = e.get('live', False)
                                            market['ended'] = e.get('ended', False)
                                            market['score'] = e.get('score')
                                            market['period'] = e.get('period')
                            except Exception:
                                pass
                            
                await asyncio.sleep(10) # Fast poll for scores
                
            except Exception as e:
                logger.error(f"Score update error: {e}")
                await asyncio.sleep(10)

    def get_websocket_subscriptions(self) -> List[str]:
        """
        Get list of token IDs for WebSocket subscription.
        Prioritizes LIVE games to stay within limits.
        """
        active_tokens = []
        
        # 1. Live/Ended Games
        for m in self.markets.values():
            if m['live'] or m['ended']:
                for t in m['tokens']:
                    active_tokens.append(t['id'])
                    
        # 2. Recent Starts (if room)
        if len(active_tokens) < 400:
            now = datetime.utcnow()
            for m in self.markets.values():
                if m['live'] or m['ended']: continue
                
                start = m.get('start_time')
                if start:
                    try:
                        from dateutil import parser
                        dt = parser.parse(start).replace(tzinfo=None)
                        diff = (now - dt).total_seconds()
                        if 0 < diff < 24 * 3600:
                            for t in m['tokens']:
                                active_tokens.append(t['id'])
                    except:
                        pass
                        
        return active_tokens[:500] # Hard cap
