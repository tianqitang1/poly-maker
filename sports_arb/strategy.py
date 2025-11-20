import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime

from poly_data.polymarket_client import PolymarketClient
from poly_utils.logging_utils import get_logger

logger = get_logger('sports_arb.strategy')

class SportsArbStrategy:
    """
    Core logic for Sports Arbitrage.
    Combines Post-Resolution Arb (Game Ended) and Flash/Momentum Arb (Live Game).
    """
    
    def __init__(self, client: PolymarketClient, config: Dict[str, Any]):
        self.client = client
        self.config = config.get('strategy', {})
        self.flash_config = self.config.get('flash_arb', {})
        self.post_res_config = self.config.get('post_res_arb', {})
        
        # State tracking
        self.executed_markets = set()
        self.price_history = {} # token_id -> previous_price
        
    async def process_market_update(self, market_info: Dict[str, Any], token_id: str, best_bid: float, best_ask: float):
        """
        Process a live price update from WebSocket.
        Main entry point for strategy execution.
        """
        market_id = market_info['condition_id']
        
        # Skip if we already traded this market
        if market_id in self.executed_markets:
            return

        # 1. Update Price History (for Flash/Momentum detection)
        prev_price = self.price_history.get(token_id)
        
        # Mid price as reference
        current_price = (best_bid + best_ask) / 2.0 if (best_bid and best_ask) else 0
        if current_price == 0: return
        
        self.price_history[token_id] = current_price
        
        # 2. Check Game Status
        is_live = market_info.get('live', False)
        is_ended = market_info.get('ended', False)
        
        if is_ended:
            # Strategy A: Post-Resolution Arb
            await self._check_post_resolution(market_info, token_id, best_ask)
            
        elif is_live and self.flash_config.get('enabled', True):
            # Strategy B: Flash/Momentum Arb
            await self._check_flash_opportunity(market_info, token_id, current_price, prev_price, best_ask)

    async def _check_post_resolution(self, market: Dict[str, Any], token_id: str, best_ask: float):
        """
        Check if game is ended and we can buy the winner cheap.
        """
        score = market.get('score', '')
        if not score or score == '0-0': return # Ignore canceled/empty
        
        # Parse Score
        try:
            parts = score.split('-')
            if len(parts) != 2: return
            s1 = int(parts[0].strip().split()[-1])
            s2 = int(parts[1].strip().split()[0])
            
            # Identify Winner (Outcome 0 or Outcome 1)
            # Convention: Team 1 is Outcome 0, Team 2 is Outcome 1
            winner_idx = 0 if s1 > s2 else 1
            if s1 == s2: return # Draw
            
            winner_token_id = market['tokens'][winner_idx]['id']
            
            # If the token we are looking at IS the winner
            if token_id == winner_token_id:
                price = best_ask
                ceiling = self.post_res_config.get('price_ceiling', 0.99)
                
                if 0.50 < price <= ceiling:
                    logger.info(f"🏆 POST-RES OPPORTUNITY: {market['question']}")
                    logger.info(f"   Winner: {market['tokens'][winner_idx]['outcome']} (Score: {score})")
                    logger.info(f"   Price: {price}")
                    await self._execute_trade(market, token_id, price, "POST_RES_ARB")
                    
        except Exception as e:
            logger.error(f"Error parsing score {score}: {e}")

    async def _check_flash_opportunity(self, market: Dict[str, Any], token_id: str, current_price: float, prev_price: Optional[float], best_ask: float):
        """
        Check for Momentum or Liquidity events in Live games.
        """
        if not prev_price: return
        
        # Params
        min_price = self.flash_config.get('min_price_to_trade', 0.90)
        momentum_thresh = self.flash_config.get('momentum_threshold', 0.02)
        
        # Logic 1: Momentum Ignition (Buying Strength)
        # If price is ALREADY high (likely winner) AND it jumps UP
        # Example: 0.92 -> 0.95. 
        # This confirms the win is solidifying.
        if current_price >= min_price:
            change = current_price - prev_price
            if change >= momentum_thresh:
                logger.info(f"🚀 MOMENTUM BUY: {market['question']}")
                logger.info(f"   Token: {token_id} | Move: {prev_price:.3f} -> {current_price:.3f}")
                await self._execute_trade(market, token_id, best_ask, "FLASH_MOMENTUM")
                
        # Logic 2: Reversal / Flip (Buying the Comeback)
        # User scenario: Favorite crashes 90c -> 40c (Don't buy favorite).
        # Wait until Favorite < 10c (meaning Underdog > 90c).
        # Then BUY UNDERDOG.
        # This is actually covered by Logic 1! 
        # If Underdog goes from 10c -> 90c, the 'current_price >= min_price' check will fail until it hits 90c.
        # Once it hits 90c, we are in "Buying Strength" territory.
        
        # What about the "Liquidity Vacuum" user mentioned? 
        # "Buy orders lower than 99c were gone, bid goes to 99.9c"
        # This is exactly what Logic 1 catches. If bid jumps 90->99, `change` is +0.09, triggers buy.

    async def _execute_trade(self, market: Dict[str, Any], token_id: str, price: float, strategy_type: str):
        """Execute the trade."""
        if self.config.get('dry_run', True):
            logger.info(f" [DRY RUN] Would BUY {token_id} @ {price} ({strategy_type})")
            self.executed_markets.add(market['condition_id'])
            return

        # Live Trade
        size = self.config.get('max_position_size', 10.0)
        try:
            logger.info(f"💸 EXECUTING BUY: {token_id} @ {price}, Size: {size}")
            resp = self.client.create_order(
                marketId=token_id,
                action="BUY",
                price=price,
                size=size
            )
            logger.info(f"   Order Response: {resp}")
            self.executed_markets.add(market['condition_id'])
        except Exception as e:
            logger.error(f"Trade failed: {e}")
