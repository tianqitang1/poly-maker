"""
Fast Execution Engine for LLM Maker Bot

Translates LLM trading signals into actual orders quickly.
Uses pre-configured logic for immediate execution while
respecting LLM guidance on direction and confidence.
"""

import sys
import os
import asyncio
import traceback
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import poly_data.global_state as global_state
from poly_data.data_utils import get_position, get_order
from poly_data.trading_utils import get_best_bid_ask_deets, round_down, round_up
from poly_utils.logging_utils import get_logger

logger = get_logger('llm_maker.execution')


class ExecutionEngine:
    """Executes trades based on LLM signals."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize execution engine.

        Args:
            config: Full configuration dict with 'execution' and 'strategy' sections
        """
        self.config = config
        self.execution_config = config.get('execution', {})
        self.strategy_config = config.get('strategy', {})

        # Active signals (updated by LLM decision loop)
        self.active_signals = {}  # market_id -> signal dict

    def update_signals(self, signals: List[Dict[str, Any]]):
        """
        Update active trading signals from LLM.

        Args:
            signals: List of signal dicts from LLM strategy
        """
        # Clear old signals
        self.active_signals = {}

        # Add new signals
        for signal in signals:
            market_id = signal.get('market_id')
            if market_id:
                self.active_signals[market_id] = signal

        logger.info(f"Updated signals for {len(self.active_signals)} markets")

    async def execute_market(self, market_id: str):
        """
        Execute trades for a specific market based on active signals.

        This is called by the WebSocket handler when market data updates.

        Args:
            market_id: Market condition_id
        """
        # Check if we have a signal for this market
        if market_id not in self.active_signals:
            return  # No signal, skip

        signal = self.active_signals[market_id]
        action = signal['action']
        confidence = signal['confidence']

        # Check confidence threshold
        min_confidence = self.strategy_config.get('min_confidence_to_trade', 0.6)
        if confidence < min_confidence:
            logger.debug(f"Skipping {market_id}: confidence {confidence} < threshold {min_confidence}")
            return

        # Get market data
        try:
            market_df = global_state.df[global_state.df['condition_id'] == market_id]
            if market_df.empty:
                logger.warning(f"Market {market_id} not found in dataframe")
                return

            row = market_df.iloc[0]
        except Exception as e:
            logger.error(f"Error getting market data for {market_id}: {e}")
            return

        # Execute based on action
        try:
            if action == 'exit':
                await self._execute_exit(row, signal)
            elif action in ['buy_yes', 'buy_no']:
                await self._execute_buy(row, signal)
            elif action in ['sell_yes', 'sell_no']:
                await self._execute_sell(row, signal)
            elif action == 'hold':
                # Just monitor, maybe adjust orders
                await self._execute_hold(row, signal)
            else:
                logger.warning(f"Unknown action: {action}")

        except Exception as e:
            logger.error(f"Error executing trade for {market_id}: {e}")
            logger.debug(traceback.format_exc())

    async def _execute_exit(self, row: Any, signal: Dict[str, Any]):
        """Exit all positions for this market."""
        client = global_state.client
        market_id = signal['market_id']

        logger.info(f"Executing EXIT for {row['question']} (reason: {signal['reasoning']})")

        # Get positions
        pos_1 = get_position(row['token1'])
        pos_2 = get_position(row['token2'])

        # Cancel all existing orders
        if pos_1['size'] > 0 or pos_2['size'] > 0:
            try:
                client.cancel_all_market(market_id)
                await asyncio.sleep(0.5)  # Give cancel time to process
            except Exception as e:
                logger.error(f"Error cancelling orders: {e}")

        # Sell all positions at market price
        for token, pos in [(row['token1'], pos_1), (row['token2'], pos_2)]:
            if pos['size'] > 10:  # Only sell if position > 10 shares
                try:
                    # Get best bid to sell at market
                    token_name = 'token1' if token == row['token1'] else 'token2'
                    deets = get_best_bid_ask_deets(market_id, token_name, 20, 0.1)

                    if deets['best_bid'] and deets['best_bid'] > 0:
                        sell_price = deets['best_bid']
                        logger.info(f"Selling {pos['size']} shares of {token} @ {sell_price}")

                        client.create_order(
                            token,
                            'SELL',
                            sell_price,
                            pos['size'],
                            row['neg_risk'] == 'TRUE'
                        )
                except Exception as e:
                    logger.error(f"Error selling position for token {token}: {e}")

    async def _execute_buy(self, row: Any, signal: Dict[str, Any]):
        """Execute buy order based on LLM signal."""
        client = global_state.client
        market_id = signal['market_id']
        action = signal['action']
        confidence = signal['confidence']
        directional_bias = signal.get('directional_bias', 'neutral')

        # Determine which token to buy
        if action == 'buy_yes':
            token = row['token1']
            token_name = 'token1'
            answer = row['answer1']
        else:  # buy_no
            token = row['token2']
            token_name = 'token2'
            answer = row['answer2']

        # Get current position
        pos = get_position(token)
        current_size = pos['size']

        # Calculate position size based on confidence
        base_size = self.strategy_config.get('base_position_size', 50)
        max_size = self.strategy_config.get('max_position_size', 200)

        # Confidence multiplier: higher confidence = larger position
        min_conf_to_size_up = self.strategy_config.get('min_confidence_to_size_up', 0.8)
        if confidence >= min_conf_to_size_up:
            multiplier = self.strategy_config.get('confidence_multiplier', 1.5)
            target_size = base_size * (1 + (confidence - min_conf_to_size_up) * multiplier)
        else:
            target_size = base_size

        target_size = min(target_size, max_size)

        # Check if we should buy more
        if current_size >= target_size * 0.95:
            logger.debug(f"Position {current_size} already at target {target_size}, skipping buy")
            return

        buy_amount = target_size - current_size

        # Ensure minimum size
        min_size = row.get('min_size', 10)
        if buy_amount < min_size:
            buy_amount = min_size

        # Get order book
        try:
            deets = get_best_bid_ask_deets(market_id, token_name, 100, 0.1)
            if not deets['best_bid'] or not deets['best_ask']:
                logger.warning(f"Insufficient market data for {market_id}")
                return
        except Exception as e:
            logger.error(f"Error getting order book: {e}")
            return

        # Calculate bid price based on directional bias
        tick_size = row['tick_size']
        tick_improvement = self.execution_config.get('tick_improvement', 1)

        if directional_bias == 'bullish':
            # More aggressive: improve price more
            bid_price = deets['best_bid'] + (tick_improvement + 1) * tick_size
        else:
            # Standard: improve by configured amount
            bid_price = deets['best_bid'] + tick_improvement * tick_size

        # Ensure bid doesn't cross spread
        if bid_price >= deets['best_ask']:
            bid_price = deets['best_bid'] + tick_size

        # Round to appropriate precision
        round_length = len(str(tick_size).split(".")[1])
        bid_price = round(bid_price, round_length)

        # Validate price range
        if not (0.10 <= bid_price <= 0.90):
            logger.warning(f"Bid price {bid_price} outside valid range, skipping")
            return

        # Check spread threshold
        spread = deets['best_ask'] - deets['best_bid']
        max_spread = self.execution_config.get('max_spread_threshold', 0.10)
        if spread > max_spread:
            logger.warning(f"Spread {spread} exceeds threshold {max_spread}, skipping")
            return

        # Place order
        logger.info(f"Buying {buy_amount} shares of {answer} @ {bid_price} (confidence: {confidence:.2f}, bias: {directional_bias})")
        logger.info(f"Reasoning: {signal['reasoning']}")

        try:
            # Cancel existing orders for this token first
            client.cancel_all_asset(token)
            await asyncio.sleep(0.3)

            # Place buy order
            client.create_order(
                token,
                'BUY',
                bid_price,
                buy_amount,
                row['neg_risk'] == 'TRUE'
            )
        except Exception as e:
            logger.error(f"Error placing buy order: {e}")

    async def _execute_sell(self, row: Any, signal: Dict[str, Any]):
        """Execute sell order based on LLM signal."""
        client = global_state.client
        market_id = signal['market_id']
        action = signal['action']

        # Determine which token to sell
        if action == 'sell_yes':
            token = row['token1']
            token_name = 'token1'
            answer = row['answer1']
        else:  # sell_no
            token = row['token2']
            token_name = 'token2'
            answer = row['answer2']

        # Get current position
        pos = get_position(token)
        if pos['size'] < 10:
            logger.debug(f"Position too small to sell: {pos['size']}")
            return

        # Sell entire position
        sell_amount = pos['size']

        # Get order book
        try:
            deets = get_best_bid_ask_deets(market_id, token_name, 100, 0.1)
            if not deets['best_bid'] or not deets['best_ask']:
                logger.warning(f"Insufficient market data for {market_id}")
                return
        except Exception as e:
            logger.error(f"Error getting order book: {e}")
            return

        # Calculate ask price
        tick_size = row['tick_size']
        tick_improvement = self.execution_config.get('tick_improvement', 1)
        ask_price = deets['best_ask'] - tick_improvement * tick_size

        # Ensure ask doesn't cross spread
        if ask_price <= deets['best_bid']:
            ask_price = deets['best_ask']

        # Round to appropriate precision
        round_length = len(str(tick_size).split(".")[1])
        ask_price = round(ask_price, round_length)

        # Place order
        logger.info(f"Selling {sell_amount} shares of {answer} @ {ask_price}")
        logger.info(f"Reasoning: {signal['reasoning']}")

        try:
            # Cancel existing orders first
            client.cancel_all_asset(token)
            await asyncio.sleep(0.3)

            # Place sell order
            client.create_order(
                token,
                'SELL',
                ask_price,
                sell_amount,
                row['neg_risk'] == 'TRUE'
            )
        except Exception as e:
            logger.error(f"Error placing sell order: {e}")

    async def _execute_hold(self, row: Any, signal: Dict[str, Any]):
        """
        Execute hold strategy: monitor position and adjust orders if needed.

        This is less aggressive than active buying/selling.
        """
        # For now, just log that we're holding
        # In future, could implement passive order placement here
        logger.debug(f"Holding position for {row['question']}: {signal['reasoning']}")
