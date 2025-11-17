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

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import re
import json

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

        # LLM usage - make it OPTIONAL since game metadata is more reliable
        self.use_llm_verification = arb_config.get('use_llm_verification', False)  # Default: OFF
        self.high_confidence_price_threshold = arb_config.get('high_confidence_price_threshold', 0.97)  # 97c+

        # Per-opportunity capital allocation
        # Controls what % of arb capital to use PER opportunity
        # Examples:
        #   1.0 = 100% (only 1 concurrent arb, small accounts)
        #   0.1 = 10% (up to 10 concurrent arbs, large accounts)
        self.per_opportunity_allocation = arb_config.get('per_opportunity_allocation', 1.0)

        # Verified results cache
        self.verified_results: Dict[str, GameResult] = {}  # market_id -> result

        # Cache for "not ended yet" markets to avoid repeated LLM calls
        # Format: {market_id: (last_checked_timestamp, game_status)}
        self.not_ended_cache: Dict[str, tuple] = {}

        # Cache for live games (should check very frequently!)
        # Format: {market_id: last_checked_timestamp}
        self.live_games: Dict[str, datetime] = {}

        # Track active positions for capital management
        # Format: {market_id: allocated_capital}
        self.active_positions: Dict[str, float] = {}

        # Executed arbs (to avoid duplicates)
        self.executed_arbs = set()  # market_ids we've already arbed

        logger.info(
            f"Initialized PostResolutionArbitrage "
            f"(enabled={self.enabled}, min_profit={self.min_profit_cents}c, "
            f"capital={self.capital_allocation*100:.0f}%, "
            f"per_opp={self.per_opportunity_allocation*100:.0f}%)"
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
            result = None

            # PRIORITY 1: Try game metadata (fast, reliable, real-time!)
            game_metadata = market_info.get('game_metadata', {})
            if game_metadata:
                # For game metadata verification, we need to estimate winning price
                # Use current_price as proxy for now
                result = self._verify_from_game_metadata(
                    market_id,
                    market_question,
                    game_metadata,
                    winning_token_price=current_price  # Will be refined later
                )

            # PRIORITY 2: LLM verification (if enabled and game metadata didn't work)
            if not result and self.use_llm_verification:
                logger.info("Game metadata verification failed, falling back to LLM...")
                result = await self._verify_game_result(
                    market_id,
                    market_question,
                    game_metadata=game_metadata,
                    market_description=market_info.get('description', ''),
                    current_price=current_price
                )

            # Store verified result
            if result and result.confidence >= self.min_confidence:
                self.verified_results[market_id] = result
                # Remove from not_ended_cache since it ended
                if market_id in self.not_ended_cache:
                    del self.not_ended_cache[market_id]
            else:
                return None  # Can't verify result yet

        result = self.verified_results[market_id]

        # Get the actual order book for the winning token
        # Need to fetch YES or NO token price depending on winner
        try:
            # Get token IDs from market_info
            yes_token = market_info.get('yes_token')
            no_token = market_info.get('no_token')

            if not yes_token or not no_token:
                logger.warning(f"Missing token IDs for market {market_id}")
                return None

            # Determine which token won
            winning_token_id = yes_token if result.winner == 'yes' else no_token

            # Fetch order book for winning token
            order_book = self.client.get_order_book(winning_token_id)

            if not order_book or 'asks' not in order_book:
                logger.warning(f"Could not fetch order book for {winning_token_id}")
                # Fall back to approximation
                winning_token_price = current_price if result.winner == 'yes' else (1 - current_price)
            else:
                # Get best ask (price we'd pay to buy)
                asks = order_book.get('asks', [])
                if not asks:
                    logger.warning(f"No asks available for {winning_token_id}")
                    return None  # Can't buy if no one is selling

                # Best ask is the lowest price someone is willing to sell at
                best_ask = float(asks[0]['price'])
                winning_token_price = best_ask

                logger.info(f"Winning token ({result.winner.upper()}) best ask: ${winning_token_price:.3f}")

        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            # Fall back to approximation
            winning_token_price = current_price if result.winner == 'yes' else (1 - current_price)

        # Check if there's arb opportunity
        expected_value = 1.00  # Winning token settles to $1.00
        profit = expected_value - winning_token_price
        profit_cents = profit * 100

        if profit_cents < self.min_profit_cents:
            return None  # Not enough profit

        if winning_token_price > self.max_price:
            return None  # Price too high (low profit margin)

        # Calculate position size with proper capital management
        total_arb_capital = self.capital_allocation * self._get_total_capital()
        allocated_capital = sum(self.active_positions.values())
        available_capital = total_arb_capital - allocated_capital

        # Per-opportunity allocation
        opportunity_capital = self.per_opportunity_allocation * total_arb_capital

        # Position size is the minimum of:
        # 1. Max position size (hard cap)
        # 2. Per-opportunity allocation
        # 3. Available capital (can't overdraw)
        position_size = min(
            self.max_position_size,
            opportunity_capital,
            available_capital
        )

        # Check if we have enough capital
        if position_size < 1.0:  # Less than $1 available
            logger.warning(
                f"Insufficient capital for arb: ${position_size:.2f} "
                f"(allocated: ${allocated_capital:.2f}/{total_arb_capital:.2f})"
            )
            return None  # Skip this opportunity

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

    def _verify_from_game_metadata(
        self,
        market_id: str,
        market_question: str,
        game_metadata: Dict[str, Any],
        winning_token_price: float
    ) -> Optional[GameResult]:
        """
        Verify game result using real-time game metadata from API.

        This is MUCH more reliable than news/LLM because:
        - Real-time updates from Polymarket's game tracking
        - No need to wait for news articles
        - Catches the arbitrage window faster

        Args:
            market_id: Market identifier
            market_question: Market question
            game_metadata: Game metadata from events array
            winning_token_price: Current price to validate signal strength

        Returns:
            GameResult or None if game not ended
        """
        if not game_metadata:
            logger.debug(f"No game metadata for {market_question}")
            return None

        ended = game_metadata.get('ended', False)
        live = game_metadata.get('live', False)
        score = game_metadata.get('score', '')
        period = game_metadata.get('period', '')
        finished_timestamp = game_metadata.get('finishedTimestamp', '')
        elapsed = game_metadata.get('elapsed', '')
        start_time = game_metadata.get('startTime', '')

        # ===== COMPREHENSIVE LOGGING FOR DATA ANALYSIS =====
        # Log ALL game metadata to understand real-world values
        logger.info(f"📊 GAME METADATA SAMPLE [{market_id[:8]}...]")
        logger.info(f"   Question: {market_question}")
        logger.info(f"   live: {live} (type: {type(live).__name__})")
        logger.info(f"   ended: {ended} (type: {type(ended).__name__})")
        logger.info(f"   score: '{score}' (type: {type(score).__name__})")
        logger.info(f"   period: '{period}' (type: {type(period).__name__})")
        logger.info(f"   elapsed: '{elapsed}' (type: {type(elapsed).__name__})")
        logger.info(f"   finishedTimestamp: '{finished_timestamp}'")
        logger.info(f"   startTime: '{start_time}'")
        logger.info(f"   Raw metadata: {json.dumps(game_metadata, indent=2)}")
        # ===================================================

        # Track live games for frequent monitoring
        if live and not ended:
            logger.info(f"🔴 LIVE GAME: {market_question} ({period}, {score})")
            self.live_games[market_id] = datetime.now()
            return None

        # Game hasn't ended yet
        if not ended:
            logger.debug(f"Game not ended: {market_question}")
            return None

        # GAME ENDED! Extract winner from score
        if not score:
            logger.warning(f"Game ended but no score: {market_question}")
            return None

        # Parse score to determine winner
        # Format: "147-150" or "Team1 147-150"
        logger.info(f"🏁 GAME ENDED: {market_question}")
        logger.info(f"   Score: {score}")
        logger.info(f"   Period: {period}")
        logger.info(f"   Finished: {finished_timestamp}")

        # Determine winner based on score and market question
        winner = self._determine_winner_from_score(market_question, score)

        if not winner:
            logger.warning(f"Could not determine winner from score: {score}")
            return None

        # Calculate confidence based on price signal
        # If winning token is trading >97c, VERY high confidence
        confidence = 99 if winning_token_price > self.high_confidence_price_threshold else 95

        result = GameResult(
            market_id=market_id,
            market_question=market_question,
            winner=winner,
            confidence=confidence,
            final_score=score,
            verification_source='game_metadata',
            reasoning=f"Game ended: {score} ({period}). Price signal: ${winning_token_price:.3f}",
            game_status='ended'
        )

        logger.info(f"✅ Verified from game metadata: {result}")

        # Remove from live games
        if market_id in self.live_games:
            del self.live_games[market_id]

        return result

    def _determine_winner_from_score(
        self,
        market_question: str,
        score: str
    ) -> Optional[str]:
        """
        Determine winner from market question and score.

        Examples:
        - "Bulls vs. Jazz", score "147-150" → Jazz wins → check if market is "Bulls" or "Jazz"
        - "Will Bulls beat Jazz?", score "147-150" → Bulls lost → winner="no"

        Returns:
            'yes' or 'no' depending on market structure
        """
        try:
            # Parse score like "147-150" or "Team1 147-150"
            score_parts = score.split('-')
            if len(score_parts) != 2:
                return None

            # Extract just the numbers
            score1_str = score_parts[0].strip().split()[-1]  # Get last token (the number)
            score2_str = score_parts[1].strip().split()[0]  # Get first token (the number)

            score1 = int(score1_str)
            score2 = int(score2_str)

            # Determine which team won
            team1_won = score1 > score2

            # Parse market question to determine market structure
            q_lower = market_question.lower()

            # Check if it's a "Will X beat Y?" style question
            if 'will' in q_lower and 'beat' in q_lower:
                # Extract team names
                # "Will Bulls beat Jazz?" → Bulls is team1
                # If Bulls won (team1_won=True), answer is YES
                return 'yes' if team1_won else 'no'
            elif 'vs' in q_lower or 'vs.' in q_lower:
                # "Bulls vs. Jazz" → First team is team1
                # Need to check which outcome is YES
                # This requires checking token outcomes...
                # For now, assume YES = first team
                return 'yes' if team1_won else 'no'
            else:
                # Unknown format
                logger.warning(f"Unknown market question format: {market_question}")
                return None

        except Exception as e:
            logger.error(f"Error parsing score '{score}': {e}")
            return None

    async def _verify_game_result(
        self,
        market_id: str,
        market_question: str,
        game_metadata: Optional[Dict[str, Any]] = None,
        market_description: Optional[str] = None,
        current_price: Optional[float] = None
    ) -> Optional[GameResult]:
        """
        Verify game result using game metadata + LLM.

        Args:
            market_id: Market identifier
            market_question: Market question
            game_metadata: Game metadata from Polymarket API (score, period, etc.)
            market_description: Market description text
            current_price: Current market price for validation

        Returns:
            GameResult or None if can't verify
        """
        logger.info(f"Verifying game result for: {market_question}")

        # Use LLM to verify result
        if not self.llm_provider:
            logger.warning("LLM provider not available for verification")
            return None

        # Build verification prompt using game metadata (MUCH better than news!)
        prompt = self._build_verification_prompt(
            market_question,
            game_metadata=game_metadata,
            market_description=market_description,
            current_price=current_price
        )

        # Log the full prompt for debugging
        logger.info(f"LLM Verification Prompt for '{market_question[:50]}...':")
        logger.info("-" * 80)
        logger.info(prompt)
        logger.info("-" * 80)

        # Call LLM
        response = self.llm_provider.analyze(prompt, json_mode=True)

        if not response['success']:
            logger.error(f"LLM verification failed: {response.get('error')}")
            return None

        # Log the full response for debugging
        logger.info(f"LLM Verification Response:")
        logger.info("-" * 80)
        logger.info(json.dumps(response.get('content', {}), indent=2))
        logger.info("-" * 80)

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
        game_metadata: Optional[Dict[str, Any]] = None,
        market_description: Optional[str] = None,
        current_price: Optional[float] = None
    ) -> str:
        """Build LLM prompt for result verification using Polymarket game metadata."""

        # Build game data section
        game_data_text = "No game metadata available"
        if game_metadata:
            game_data_text = f"""- Live: {game_metadata.get('live', 'unknown')}
- Ended: {game_metadata.get('ended', 'unknown')}
- Score: {game_metadata.get('score', 'N/A')}
- Period: {game_metadata.get('period', 'N/A')}
- Elapsed Time: {game_metadata.get('elapsed', 'N/A')}
- Finished Timestamp: {game_metadata.get('finishedTimestamp', 'N/A')}
- Start Time: {game_metadata.get('startTime', 'N/A')}"""

        # Add market description if available
        description_text = ""
        if market_description:
            description_text = f"""
MARKET DESCRIPTION:
{market_description}
"""

        # Add price signal if available
        price_signal_text = ""
        if current_price is not None:
            price_signal_text = f"""
CURRENT MARKET PRICE:
- YES token: ${current_price:.3f} ({current_price*100:.1f}% probability)
- NO token: ${1-current_price:.3f} ({(1-current_price)*100:.1f}% probability)
"""

        prompt = f"""You are verifying the outcome of a sports betting market using REAL-TIME game data from Polymarket.

MARKET QUESTION: "{market_question}"
{description_text}{price_signal_text}
GAME DATA (from Polymarket API - REAL-TIME):
{game_data_text}

VERIFICATION TASK:
Analyze the game data to determine:
1. What is the current status of the game? (not_started, live, ended, unknown)
2. Has the game concluded? (game_ended: true/false)
3. If ended, what was the final score and who won?
4. Based on the score and market question, should this resolve to YES or NO?
5. How confident are you in this determination (0-100%)?

OUTPUT FORMAT (JSON):
{{
  "game_status": "not_started" | "live" | "ended" | "unknown",
  "game_ended": true or false,
  "winner": "yes" or "no" (ONLY if game_ended=true),
  "final_score": "e.g., Jazz 150-147 (parsed from score field)",
  "confidence": 0-100,
  "reasoning": "brief explanation of your analysis"
}}

INTERPRETATION GUIDE:
- game_status = "ended" AND ended = true → Game is COMPLETELY OVER
- game_status = "live" OR live = true → Game is IN PROGRESS
- Score format is typically "Team1Score-Team2Score" (e.g., "147-150")
- Period examples: "Q4" (4th quarter), "VFT" (Final OT), "F" (Final)

CRITICAL RULES:
1. If ended=true, the game is OVER - parse the score to determine winner
2. For "Will X beat Y?" markets: winner="yes" if X won, winner="no" if Y won
3. For "X vs Y" markets: Determine which team is YES outcome, then check who won
4. Score format: Higher score wins (parse carefully, e.g., "147-150" means 147 < 150)
5. High confidence (95+) if game ended=true AND score is available
6. Medium confidence (75-90) if strong price signal but metadata unclear
7. Low confidence (<75) if data is ambiguous or incomplete

EXAMPLES:
- Question: "Will Bulls beat Jazz?", Score: "147-150", Ended: true
  → Bulls scored 147, Jazz scored 150, Jazz won, winner="no" (Bulls did NOT beat Jazz)

- Question: "Bulls vs Jazz", Score: "147-150", Ended: true
  → Need to determine which team is YES - typically first mentioned team
  → Bulls scored 147 (lost), winner="no"

Respond with valid JSON only."""

        return prompt

    def _get_total_capital(self) -> float:
        """Get total capital for arb strategy."""
        # TODO: Integrate with actual account balance
        # For now, return configured total capital
        total_capital = self.config.get('capital', {}).get('total_capital', 1000.0)
        return total_capital

    def release_capital(self, market_id: str) -> None:
        """
        Release allocated capital when market resolves.

        Call this after a market settles to free up capital for new opportunities.

        Args:
            market_id: Market that resolved
        """
        if market_id in self.active_positions:
            released = self.active_positions.pop(market_id)
            logger.info(f"Released ${released:.2f} from resolved market {market_id[:8]}...")

            total_arb_capital = self.capital_allocation * self._get_total_capital()
            allocated_capital = sum(self.active_positions.values())
            available_capital = total_arb_capital - allocated_capital

            logger.info(
                f"Capital status: ${available_capital:.2f} available / "
                f"${allocated_capital:.2f} allocated / ${total_arb_capital:.2f} total"
            )

    def get_capital_status(self) -> Dict[str, float]:
        """Get current capital allocation status."""
        total_arb_capital = self.capital_allocation * self._get_total_capital()
        allocated_capital = sum(self.active_positions.values())
        available_capital = total_arb_capital - allocated_capital

        return {
            'total': total_arb_capital,
            'allocated': allocated_capital,
            'available': available_capital,
            'active_positions': len(self.active_positions),
            'utilization': allocated_capital / total_arb_capital if total_arb_capital > 0 else 0
        }

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

        # LIVE EXECUTION
        try:
            # Get market info to determine token and neg_risk status
            market_info = self.client.get_market(market_id)
            if not market_info:
                logger.error(f"Could not fetch market info for {market_id}")
                print(f"\n❌ ERROR: Could not fetch market info")
                print(f"{'='*120}\n")
                return {
                    'success': False,
                    'error': 'Could not fetch market info',
                    'opportunity': opportunity
                }

            # Get winning token ID
            tokens = market_info.get('tokens', [])
            if len(tokens) < 2:
                logger.error(f"Invalid token structure for market {market_id}")
                print(f"\n❌ ERROR: Invalid token structure")
                print(f"{'='*120}\n")
                return {
                    'success': False,
                    'error': 'Invalid token structure',
                    'opportunity': opportunity
                }

            # YES token is typically tokens[0], NO is tokens[1]
            yes_token_id = tokens[0].get('token_id', '')
            no_token_id = tokens[1].get('token_id', '')

            winning_token_id = yes_token_id if opportunity['winner'] == 'yes' else no_token_id

            # Check if neg_risk market
            neg_risk = market_info.get('neg_risk', False)

            # Calculate buy size (dollar amount)
            buy_size = opportunity['position_size']
            buy_price = opportunity['current_price']

            logger.info(f"Placing BUY order: {winning_token_id} @ ${buy_price:.4f}, size: ${buy_size:.2f}")
            print(f"\n📝 Placing order...")
            print(f"   Token: {opportunity['winner'].upper()}")
            print(f"   Price: ${buy_price:.4f}")
            print(f"   Size: ${buy_size:.2f}")

            # Place the order
            response = self.client.create_order(
                marketId=winning_token_id,
                action='BUY',
                price=buy_price,
                size=buy_size,
                neg_risk=neg_risk
            )

            if response:
                order_id = response.get('orderID', 'N/A') if isinstance(response, dict) else 'N/A'

                # Track allocated capital for this position
                self.active_positions[market_id] = buy_size

                # Calculate remaining capital
                total_arb_capital = self.capital_allocation * self._get_total_capital()
                allocated_capital = sum(self.active_positions.values())
                available_capital = total_arb_capital - allocated_capital

                print(f"\n✅ ORDER PLACED SUCCESSFULLY!")
                print(f"   Order ID: {order_id}")
                print(f"   Expected Profit: ${opportunity['estimated_profit']:.2f} (when market resolves)")
                print(f"   Capital Allocated: ${buy_size:.2f}")
                print(f"   Capital Remaining: ${available_capital:.2f} / ${total_arb_capital:.2f}")
                print(f"   Active Positions: {len(self.active_positions)}")
                print(f"{'='*120}\n")

                # Mark as executed to avoid duplicates
                self.executed_arbs.add(market_id)

                logger.info(
                    f"Post-res arb executed successfully: {order_id} "
                    f"(allocated: ${buy_size:.2f}, remaining: ${available_capital:.2f})"
                )

                return {
                    'success': True,
                    'order_id': order_id,
                    'opportunity': opportunity,
                    'response': response
                }
            else:
                print(f"\n❌ ORDER FAILED (empty response)")
                print(f"{'='*120}\n")
                logger.error(f"Order placement failed for {market_id}")
                return {
                    'success': False,
                    'error': 'Order placement failed (empty response)',
                    'opportunity': opportunity
                }

        except Exception as e:
            logger.error(f"Error executing arb: {e}")
            import traceback
            traceback.print_exc()

            print(f"\n❌ ERROR EXECUTING TRADE: {e}")
            print(f"{'='*120}\n")

            return {
                'success': False,
                'error': str(e),
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
