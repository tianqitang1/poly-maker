"""
Post-Resolution Arbitrage

Detects when sports games end and sweeps underpriced winning tokens
before market officially settles.

Strategy:
1. Monitor for game completion (final score, game ended)
2. Verify winner with LLM + news sources
3. Sweep order book for winning token < $1.00
4. Wait for settlement (instant profit)

Risk: Very low (game is over, winner is known)
Profit: 1-5c per share (but risk-free!)
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import re

from spike_momentum.news_monitor import SportsNewsMonitor, NewsItem
from spike_momentum.llm_provider import LLMProvider
from poly_data.polymarket_client import PolymarketClient
from poly_utils.logging_utils import get_logger

logger = get_logger('spike_momentum.post_res_arb')


class GameResult:
    """Represents a verified game result."""

    def __init__(
        self,
        market_id: str,
        market_question: str,
        winner: str,  # 'yes' or 'no'
        confidence: int,  # 0-100
        final_score: Optional[str] = None,
        verification_source: str = 'llm',
        reasoning: str = '',
        game_status: str = 'unknown'  # 'not_started', 'live', 'ended', 'unknown'
    ):
        self.market_id = market_id
        self.market_question = market_question
        self.winner = winner
        self.confidence = confidence
        self.final_score = final_score
        self.verification_source = verification_source
        self.reasoning = reasoning
        self.game_status = game_status
        self.verified_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'market_id': self.market_id,
            'market_question': self.market_question,
            'winner': self.winner,
            'confidence': self.confidence,
            'final_score': self.final_score,
            'verification_source': self.verification_source,
            'reasoning': self.reasoning,
            'game_status': self.game_status,
            'verified_at': self.verified_at.isoformat()
        }

    def __repr__(self) -> str:
        return f"<GameResult: {self.market_question[:40]} status={self.game_status} winner={self.winner} conf={self.confidence}%>"


class PostResolutionArbitrage:
    """Detects and executes post-resolution arbitrage opportunities."""

    def __init__(
        self,
        client: PolymarketClient,
        news_monitor: SportsNewsMonitor,
        llm_provider: Optional[LLMProvider] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize post-resolution arbitrage.

        Args:
            client: Polymarket client for executing trades
            news_monitor: News monitor for game results
            llm_provider: LLM for result verification
            config: Configuration dict
        """
        self.client = client
        self.news_monitor = news_monitor
        self.llm_provider = llm_provider
        self.config = config or {}

        # Get config values
        arb_config = self.config.get('post_resolution_arb', {})
        self.enabled = arb_config.get('enabled', False)
        self.min_profit_cents = arb_config.get('min_profit_cents', 0.5)  # 0.5c minimum
        self.max_price = arb_config.get('max_purchase_price', 0.995)  # Don't pay more than 99.5c
        self.min_confidence = arb_config.get('min_confidence', 90)  # 90% confidence to execute
        self.max_position_size = arb_config.get('max_position_size', 100)  # Max $ per arb
        self.capital_allocation = arb_config.get('capital_allocation', 0.3)  # 30% of total capital

        # Verified results cache
        self.verified_results: Dict[str, GameResult] = {}  # market_id -> result

        # Cache for "not ended yet" markets to avoid repeated LLM calls
        # Format: {market_id: (last_checked_timestamp, game_status)}
        self.not_ended_cache: Dict[str, tuple] = {}

        # Cache for live games (should check very frequently!)
        # Format: {market_id: last_checked_timestamp}
        self.live_games: Dict[str, datetime] = {}

        # Executed arbs (to avoid duplicates)
        self.executed_arbs = set()  # market_ids we've already arbed

        logger.info(
            f"Initialized PostResolutionArbitrage "
            f"(enabled={self.enabled}, min_profit={self.min_profit_cents}c, "
            f"capital={self.capital_allocation*100:.0f}%)"
        )

    async def check_market_for_arb(
        self,
        market_id: str,
        market_question: str,
        current_price: float,
        market_info: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Check if market is post-resolution arb opportunity.

        Args:
            market_id: Market identifier
            market_question: Market question
            current_price: Current mid price
            market_info: Full market metadata

        Returns:
            Arb opportunity dict or None
        """
        if not self.enabled:
            return None

        # Skip if already executed
        if market_id in self.executed_arbs:
            return None

        # Check if market is near-close or worth checking
        end_date_iso = market_info.get('end_date', '')
        if not self._should_check_market(market_id, end_date_iso):
            return None

        # Check if we have a verified result for this market
        if market_id not in self.verified_results:
            # Try to verify result
            result = await self._verify_game_result(market_id, market_question)
            if result and result.confidence >= self.min_confidence:
                self.verified_results[market_id] = result
                # Remove from not_ended_cache since it ended
                if market_id in self.not_ended_cache:
                    del self.not_ended_cache[market_id]
            else:
                return None  # Can't verify result yet

        result = self.verified_results[market_id]

        # Get current price of winning token
        # If winner is 'yes', check YES token price
        # If winner is 'no', check NO token price

        # For now, use current_price as approximation
        # In production, we'd fetch order book for the specific winning token
        winning_token_price = current_price if result.winner == 'yes' else (1 - current_price)

        # Check if there's arb opportunity
        expected_value = 1.00  # Winning token settles to $1.00
        profit = expected_value - winning_token_price
        profit_cents = profit * 100

        if profit_cents < self.min_profit_cents:
            return None  # Not enough profit

        if winning_token_price > self.max_price:
            return None  # Price too high (low profit margin)

        # Calculate position size
        position_size = min(
            self.max_position_size,
            self.capital_allocation * self._get_available_capital()
        )

        # Arb opportunity found!
        opportunity = {
            'market_id': market_id,
            'market_question': market_question,
            'winner': result.winner,
            'current_price': winning_token_price,
            'expected_value': expected_value,
            'profit_per_share': profit,
            'profit_cents': profit_cents,
            'confidence': result.confidence,
            'final_score': result.final_score,
            'position_size': position_size,
            'max_shares': int(position_size / winning_token_price),
            'estimated_profit': position_size * (profit / winning_token_price)
        }

        logger.info(f"Post-resolution arb opportunity: {opportunity}")
        return opportunity

    def _should_check_market(self, market_id: str, end_date_iso: str) -> bool:
        """
        Determine if we should check this market based on end date, game status, and cache.

        Strategy:
        - LIVE GAMES: Check every 2 minutes (game in progress!)
        - Markets past end_date: Check every 5 minutes (arb opportunity!)
        - Markets ending <24 hours: Check every 30 minutes
        - Markets ending 1-7 days: Check every 6 hours
        - Markets ending >7 days: Check every 24 hours
        - Markets >30 days away: Skip entirely

        Args:
            market_id: Market identifier
            end_date_iso: ISO format end date (e.g., "2026-12-31T23:59:59Z")

        Returns:
            True if should check, False if should skip
        """
        now = datetime.now()

        # PRIORITY 1: Live games - check very frequently!
        if market_id in self.live_games:
            last_checked = self.live_games[market_id]
            time_since_check = (now - last_checked).total_seconds()

            # Check live games every 2 minutes
            if time_since_check < 2 * 60:  # 2 minutes
                logger.debug(f"Skipping LIVE game {market_id[:8]}... (checked {time_since_check/60:.1f}m ago)")
                return False
            else:
                logger.info(f"Checking LIVE game: {market_id[:8]}")
                return True

        # Parse end date if available
        end_date = None
        if end_date_iso:
            try:
                # Handle both formats: "2026-12-31T23:59:59Z" and "2026-12-31T23:59:59.000000Z"
                end_date_iso = end_date_iso.replace('Z', '+00:00')
                from dateutil import parser
                end_date = parser.parse(end_date_iso).replace(tzinfo=None)
            except Exception as e:
                logger.debug(f"Could not parse end_date '{end_date_iso}': {e}")

        # Check cache for "not ended yet" markets
        if market_id in self.not_ended_cache:
            last_checked, game_status = self.not_ended_cache[market_id]
            time_since_check = (now - last_checked).total_seconds()

            if end_date:
                time_until_end = (end_date - now).total_seconds()

                # PRIORITY 2: Market ended but not resolved - check frequently for arb!
                if time_until_end <= 0:
                    if time_since_check < 5 * 60:  # 5 minutes
                        logger.debug(f"Skipping POST-GAME {market_id[:8]}... (checked {time_since_check/60:.1f}m ago)")
                        return False
                    else:
                        logger.info(f"Checking POST-GAME market: {market_id[:8]}")
                        return True

                # >7 days away: check every 24 hours
                if time_until_end > 7 * 24 * 3600:
                    if time_since_check < 24 * 3600:  # 24 hours
                        logger.debug(f"Skipping {market_id[:8]}... (ends in {time_until_end/86400:.1f} days, checked {time_since_check/3600:.1f}h ago)")
                        return False

                # 1-7 days away: check every 6 hours
                elif time_until_end > 24 * 3600:
                    if time_since_check < 6 * 3600:  # 6 hours
                        logger.debug(f"Skipping {market_id[:8]}... (ends in {time_until_end/3600:.1f} hours, checked {time_since_check/3600:.1f}h ago)")
                        return False

                # <24 hours: check every 30 minutes
                else:
                    if time_since_check < 30 * 60:  # 30 minutes
                        logger.debug(f"Skipping {market_id[:8]}... (ends in {time_until_end/3600:.1f} hours, checked {time_since_check/60:.1f}m ago)")
                        return False
            else:
                # No end_date, use conservative check frequency (6 hours)
                if time_since_check < 6 * 3600:
                    logger.debug(f"Skipping {market_id[:8]}... (no end_date, checked {time_since_check/3600:.1f}h ago)")
                    return False

        # If we have end_date and it's >30 days away, skip entirely
        if end_date:
            time_until_end = (end_date - now).total_seconds()
            if time_until_end > 30 * 24 * 3600:  # >30 days
                logger.debug(f"Skipping {market_id[:8]}... (ends in {time_until_end/86400:.0f} days - too far away)")
                # Cache it so we don't keep checking
                self.not_ended_cache[market_id] = (now, 'not_started')
                return False

        # Should check
        return True

    async def _verify_game_result(
        self,
        market_id: str,
        market_question: str
    ) -> Optional[GameResult]:
        """
        Verify game result using news + LLM.

        Args:
            market_id: Market identifier
            market_question: Market question

        Returns:
            GameResult or None if can't verify
        """
        logger.info(f"Verifying game result for: {market_question}")

        # Get recent news about this game
        news_matches = self.news_monitor.match_to_market(market_question, max_results=5)

        if not news_matches:
            logger.warning(f"No news found for {market_question}")
            return None

        # Use LLM to verify result
        if not self.llm_provider:
            logger.warning("LLM provider not available for verification")
            return None

        # Build verification prompt
        prompt = self._build_verification_prompt(market_question, news_matches)

        # Call LLM
        response = self.llm_provider.analyze(prompt, json_mode=True)

        if not response['success']:
            logger.error(f"LLM verification failed: {response.get('error')}")
            return None

        # Parse response
        try:
            data = response['content']

            # Check game status
            game_status = data.get('game_status', 'unknown')
            game_ended = data.get('game_ended', False)

            # Handle live games
            if game_status == 'live':
                logger.info(f"🔴 LIVE GAME: {market_question}")
                self.live_games[market_id] = datetime.now()
                # Remove from not_ended_cache if it was there
                if market_id in self.not_ended_cache:
                    del self.not_ended_cache[market_id]
                return None

            # Game not ended yet
            if not game_ended:
                logger.info(f"Game not ended yet: {market_question} (status: {game_status})")
                # Cache this result to avoid repeated checks
                self.not_ended_cache[market_id] = (datetime.now(), game_status)
                # Remove from live games if it was there
                if market_id in self.live_games:
                    del self.live_games[market_id]
                return None

            if 'winner' not in data or 'confidence' not in data:
                logger.error(f"Invalid LLM response format: {data}")
                return None

            result = GameResult(
                market_id=market_id,
                market_question=market_question,
                winner=data['winner'].lower(),  # 'yes' or 'no'
                confidence=data['confidence'],
                final_score=data.get('final_score'),
                verification_source='llm+news',
                reasoning=data.get('reasoning', ''),
                game_status='ended'
            )

            logger.info(f"✅ Verified result: {result}")

            # Remove from live games cache (game is over!)
            if market_id in self.live_games:
                del self.live_games[market_id]

            return result

        except Exception as e:
            logger.error(f"Error parsing LLM verification: {e}")
            return None

    def _build_verification_prompt(
        self,
        market_question: str,
        news_matches: List[Dict[str, Any]]
    ) -> str:
        """Build LLM prompt for result verification."""

        # Format news
        news_text = "\n".join([
            f"{i+1}. [{match['news'].source}] {match['news'].title}\n   {match['news'].summary[:200]}"
            for i, match in enumerate(news_matches[:5])
        ])

        prompt = f"""You are verifying the outcome of a sports betting market.

MARKET QUESTION: "{market_question}"

RECENT NEWS:
{news_text}

VERIFICATION TASK:
1. What is the current status of the game/event?
2. Has the game/event concluded?
3. If yes, what was the outcome?
4. Based on the news, should the market resolve to YES or NO?
5. How confident are you (0-100%)?

OUTPUT FORMAT (JSON):
{{
  "game_status": "not_started" | "live" | "ended" | "unknown",
  "game_ended": true or false,
  "winner": "yes" or "no" (which side of the market won - ONLY if game_ended=true),
  "final_score": "e.g., Jazz 150-147" (if available),
  "confidence": 0-100 (how confident you are in this result),
  "reasoning": "brief explanation"
}}

GAME STATUS DEFINITIONS:
- "not_started": Game hasn't begun yet (no live action, future date)
- "live": Game is IN PROGRESS (look for: "Q1", "Q2", "halftime", "3rd quarter", "bottom 9th", "LIVE", "ongoing", "in progress")
- "ended": Game is COMPLETELY OVER (look for: "Final", "FT", "Game Over", "final score", "wins", "defeats", "beat")
- "unknown": Cannot determine status from news

IMPORTANT:
- Only mark game_ended=true if you're certain the game is COMPLETELY over (not halftime, not intermission)
- If news shows live action (quarters, innings, periods), mark game_status="live" and game_ended=false
- For "Will X beat Y?" markets: winner="yes" if X won, winner="no" if Y won
- Be conservative - only high confidence (90+) if result is clear and verified by multiple sources
- Look for keywords like "FINAL", "FT", "defeats", "wins" to confirm game ended

Respond with valid JSON only."""

        return prompt

    def _get_available_capital(self) -> float:
        """Get available capital for arb (placeholder)."""
        # TODO: Integrate with actual account balance
        # For now, return a reasonable default
        return 1000.0  # $1000 allocated to post-res arb

    async def execute_arb(
        self,
        opportunity: Dict[str, Any],
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Execute post-resolution arbitrage.

        Args:
            opportunity: Arb opportunity dict
            dry_run: If True, simulate without executing

        Returns:
            Execution result dict
        """
        market_id = opportunity['market_id']

        logger.info(f"Executing post-res arb: {opportunity}")

        print(f"\n{'='*120}")
        print(f"💰 POST-RESOLUTION ARBITRAGE")
        print(f"{'='*120}")
        print(f"Market: {opportunity['market_question']}")
        print(f"Winner: {opportunity['winner'].upper()}")
        print(f"Final Score: {opportunity.get('final_score', 'N/A')}")
        print(f"Current Price: {opportunity['current_price']:.3f}")
        print(f"Expected Value: {opportunity['expected_value']:.2f}")
        print(f"Profit per Share: {opportunity['profit_cents']:.1f}c")
        print(f"Position Size: ${opportunity['position_size']:.2f}")
        print(f"Max Shares: {opportunity['max_shares']}")
        print(f"Estimated Profit: ${opportunity['estimated_profit']:.2f}")
        print(f"Confidence: {opportunity['confidence']}%")

        if dry_run:
            print(f"\n🔄 DRY RUN - No actual trade executed")
            print(f"{'='*120}\n")
            return {
                'success': True,
                'dry_run': True,
                'opportunity': opportunity
            }

        # TODO: Implement actual order execution
        # 1. Get order book for winning token
        # 2. Sweep asks up to max_price
        # 3. Verify fills
        # 4. Track position

        print(f"\n⚠️  LIVE EXECUTION NOT YET IMPLEMENTED")
        print(f"   Would buy {opportunity['winner']} token up to {self.max_price:.3f}")
        print(f"{'='*120}\n")

        # Mark as executed to avoid duplicates
        self.executed_arbs.add(market_id)

        return {
            'success': False,
            'error': 'Live execution not implemented yet',
            'opportunity': opportunity
        }


# Example usage
if __name__ == '__main__':
    import asyncio
    import yaml
    from spike_momentum.llm_provider import LLMProvider
    from poly_data.polymarket_client import PolymarketClient

    async def test_post_res_arb():
        # Load config
        with open('spike_momentum/config.yaml') as f:
            config = yaml.safe_load(f)

        # Initialize components
        client = PolymarketClient()
        news_monitor = SportsNewsMonitor(config.get('news', {}))
        llm_provider = LLMProvider(config['llm'])

        # Initialize arb detector
        arb = PostResolutionArbitrage(
            client=client,
            news_monitor=news_monitor,
            llm_provider=llm_provider,
            config=config
        )

        # Fetch news
        news_monitor.fetch_news(max_items=20)

        # Test with a hypothetical market
        market_question = "Will the Jazz beat the Bulls on December 20, 2024?"

        # Check for arb opportunity
        opportunity = await arb.check_market_for_arb(
            market_id='test_market_123',
            market_question=market_question,
            current_price=0.98,  # Jazz trading at 98c
            market_info={}
        )

        if opportunity:
            print("Arb opportunity found!")
            # Execute in dry-run mode
            result = await arb.execute_arb(opportunity, dry_run=True)
            print(result)
        else:
            print("No arb opportunity (game not ended or already executed)")

    asyncio.run(test_post_res_arb())
