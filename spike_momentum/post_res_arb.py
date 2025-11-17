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
        reasoning: str = ''
    ):
        self.market_id = market_id
        self.market_question = market_question
        self.winner = winner
        self.confidence = confidence
        self.final_score = final_score
        self.verification_source = verification_source
        self.reasoning = reasoning
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
            'verified_at': self.verified_at.isoformat()
        }

    def __repr__(self) -> str:
        return f"<GameResult: {self.market_question[:40]} winner={self.winner} conf={self.confidence}%>"


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

        # Check if we have a verified result for this market
        if market_id not in self.verified_results:
            # Try to verify result
            result = await self._verify_game_result(market_id, market_question)
            if result and result.confidence >= self.min_confidence:
                self.verified_results[market_id] = result
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

            # Validate required fields
            if 'game_ended' not in data or not data['game_ended']:
                logger.info(f"Game not ended yet: {market_question}")
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
                reasoning=data.get('reasoning', '')
            )

            logger.info(f"Verified result: {result}")
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
1. Has the game/event referenced in the market question concluded?
2. If yes, what was the outcome?
3. Based on the news, should the market resolve to YES or NO?
4. How confident are you (0-100%)?

OUTPUT FORMAT (JSON):
{{
  "game_ended": true or false,
  "winner": "yes" or "no" (which side of the market won),
  "final_score": "e.g., Jazz 150-147" (if available),
  "confidence": 0-100 (how confident you are in this result),
  "reasoning": "brief explanation"
}}

IMPORTANT:
- Only mark game_ended=true if you're certain the game is COMPLETELY over
- If news is unclear or game is still in progress, mark game_ended=false
- For "Will X beat Y?" markets: winner="yes" if X won, winner="no" if Y won
- Be conservative - only high confidence (90+) if result is clear and verified by multiple sources

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
