"""
Risk Manager for Arbitrage Execution

Handles partial fill scenarios and risk mitigation strategies.
"""

import time
from typing import Dict, Tuple
from datetime import datetime
from neg_risk_arb.arbitrage_scanner import load_config
from poly_utils.logging_utils import get_logger


class ArbitrageRiskManager:
    """
    Manages risk for arbitrage execution, especially partial fills.
    """

    def __init__(self, client, config_path='neg_risk_arb/config.yaml', logger=None):
        """
        Initialize the risk manager.

        Args:
            client: PolymarketClient instance
            config_path: Path to configuration file
            logger: BotLogger instance (optional)
        """
        self.client = client

        # Load configuration
        self.config = load_config(config_path)

        # Setup logging
        self.logger = logger if logger else get_logger('neg_risk_arb')

        # Track execution history
        self.execution_history = []
        self.daily_pnl = 0.0
        self.failed_markets = {}  # market_slug -> timestamp

    def handle_partial_fill(
        self,
        opportunity: Dict,
        yes_pos: float,
        no_pos: float,
        expected_size: float,
        dry_run: bool = False
    ) -> Dict:
        """
        Handle scenario where only one side filled.

        Args:
            opportunity: Opportunity dictionary
            yes_pos: Current YES position
            no_pos: Current NO position
            expected_size: Expected fill size
            dry_run: If True, simulate without executing

        Returns:
            Result dictionary
        """
        strategy = self.config.get('partial_fill_strategy', 'market_rescue')

        print(f"\n{'⚠️ '*40}")
        print(f"PARTIAL FILL DETECTED")
        print(f"YES Position: {yes_pos:.2f}, NO Position: {no_pos:.2f}")
        print(f"Expected: {expected_size:.2f}")
        print(f"Strategy: {strategy}")
        print(f"{'⚠️ '*40}\n")

        if yes_pos > 0 and no_pos == 0:
            # Only YES filled
            return self._rescue_partial_fill(
                opportunity=opportunity,
                filled_side='YES',
                filled_token=opportunity['yes_token'],
                unfilled_token=opportunity['no_token'],
                filled_size=yes_pos,
                strategy=strategy,
                dry_run=dry_run
            )

        elif no_pos > 0 and yes_pos == 0:
            # Only NO filled
            return self._rescue_partial_fill(
                opportunity=opportunity,
                filled_side='NO',
                filled_token=opportunity['no_token'],
                unfilled_token=opportunity['yes_token'],
                filled_size=no_pos,
                strategy=strategy,
                dry_run=dry_run
            )

        elif yes_pos == 0 and no_pos == 0:
            # Neither filled
            print("Neither side filled - opportunity lost")
            return {'success': False, 'error': 'Neither side filled'}

        else:
            # Both partially filled but not equal
            print(f"Unequal fills - will merge {min(yes_pos, no_pos):.2f}")
            return {'success': True, 'partial': True, 'can_merge': True}

    def _rescue_partial_fill(
        self,
        opportunity: Dict,
        filled_side: str,
        filled_token: str,
        unfilled_token: str,
        filled_size: float,
        strategy: str,
        dry_run: bool = False
    ) -> Dict:
        """
        Execute rescue strategy for partial fill.

        Args:
            opportunity: Opportunity dictionary
            filled_side: 'YES' or 'NO'
            filled_token: Token ID that filled
            unfilled_token: Token ID that didn't fill
            filled_size: Size of filled position
            strategy: Rescue strategy to use
            dry_run: If True, simulate

        Returns:
            Result dictionary
        """
        print(f"Attempting to rescue {filled_side} position of {filled_size:.2f} shares...")

        if strategy == 'market_rescue':
            return self._market_rescue(
                opportunity, unfilled_token, filled_size, filled_side, dry_run
            )

        elif strategy == 'limit_rescue':
            return self._limit_rescue(
                opportunity, unfilled_token, filled_size, filled_side, dry_run
            )

        elif strategy == 'exit_position':
            return self._exit_position(
                opportunity, filled_token, filled_size, filled_side, dry_run
            )

        else:
            print(f"Unknown strategy: {strategy}")
            return {'success': False, 'error': f'Unknown strategy: {strategy}'}

    def _market_rescue(
        self,
        opportunity: Dict,
        unfilled_token: str,
        size: float,
        filled_side: str,
        dry_run: bool
    ) -> Dict:
        """
        Rescue by placing market order on unfilled side.

        Args:
            opportunity: Opportunity dictionary
            unfilled_token: Token to buy
            size: Size to buy
            filled_side: Which side already filled
            dry_run: If True, simulate

        Returns:
            Result dictionary
        """
        # Get current ask price
        _, asks = self.client.get_order_book(unfilled_token)

        if asks.empty:
            return {'success': False, 'error': 'No liquidity on unfilled side'}

        current_ask = float(asks.iloc[-1]['price'])

        # Calculate effective total cost
        if filled_side == 'YES':
            total_cost = opportunity['yes_ask_price'] + current_ask
        else:
            total_cost = current_ask + opportunity['no_ask_price']

        # Check if loss is acceptable
        max_loss = self.config.get('max_rescue_loss_pct', 0.01)
        loss_pct = (total_cost - 1.0) / 1.0

        print(f"Rescue cost: ${current_ask:.4f}")
        print(f"Total cost: ${total_cost:.4f}")
        print(f"Loss: {loss_pct*100:.2f}% (max: {max_loss*100:.2f}%)")

        if loss_pct > max_loss:
            print(f"❌ Loss too high - switching to exit strategy")
            filled_token = opportunity['yes_token'] if filled_side == 'YES' else opportunity['no_token']
            return self._exit_position(opportunity, filled_token, size, filled_side, dry_run)

        if dry_run:
            print("[DRY RUN] Market rescue not executed")
            return {
                'success': True,
                'strategy': 'market_rescue',
                'rescue_price': current_ask,
                'accepted_loss': loss_pct
            }

        # Place market order
        print(f"Placing market rescue order: {size:.2f} @ ${current_ask:.4f}...")

        response = self.client.create_order(
            marketId=unfilled_token,
            action='BUY',
            price=current_ask,
            size=size,
            neg_risk=opportunity['neg_risk']
        )

        if response:
            print("✓ Rescue order placed")
            return {
                'success': True,
                'strategy': 'market_rescue',
                'rescue_price': current_ask,
                'accepted_loss': loss_pct,
                'order_response': response
            }
        else:
            print("❌ Rescue order failed")
            return {'success': False, 'error': 'Rescue order failed'}

    def _limit_rescue(
        self,
        opportunity: Dict,
        unfilled_token: str,
        size: float,
        filled_side: str,
        dry_run: bool
    ) -> Dict:
        """
        Rescue with limit order, fallback to market if timeout.

        Args:
            opportunity: Opportunity dictionary
            unfilled_token: Token to buy
            size: Size to buy
            filled_side: Which side filled
            dry_run: If True, simulate

        Returns:
            Result dictionary
        """
        # Get best bid price (passive)
        bids, asks = self.client.get_order_book(unfilled_token)

        if bids.empty or asks.empty:
            return {'success': False, 'error': 'No liquidity'}

        # Try to place at best bid (join the queue)
        best_bid = float(bids.iloc[-1]['price'])
        best_ask = float(asks.iloc[-1]['price'])

        print(f"Attempting limit rescue at ${best_bid:.4f}...")

        if dry_run:
            print("[DRY RUN] Limit rescue not executed")
            return {'success': True, 'strategy': 'limit_rescue'}

        # Place limit order
        response = self.client.create_order(
            marketId=unfilled_token,
            action='BUY',
            price=best_bid,
            size=size,
            neg_risk=opportunity['neg_risk']
        )

        if not response:
            print("Limit order failed - trying market rescue")
            return self._market_rescue(opportunity, unfilled_token, size, filled_side, dry_run)

        # Wait for fill
        timeout = self.config.get('rescue_timeout_sec', 5)
        print(f"Waiting {timeout}s for limit order to fill...")
        time.sleep(timeout)

        # Check if filled
        _, pos = self.client.get_position(unfilled_token)

        if pos >= size * 0.95:
            print("✓ Limit rescue successful")
            return {'success': True, 'strategy': 'limit_rescue', 'price': best_bid}
        else:
            # Cancel limit order and go to market
            print("Limit order didn't fill - switching to market")
            self.client.cancel_all_asset(unfilled_token)
            time.sleep(0.5)
            return self._market_rescue(opportunity, unfilled_token, size, filled_side, dry_run)

    def _exit_position(
        self,
        opportunity: Dict,
        filled_token: str,
        size: float,
        filled_side: str,
        dry_run: bool
    ) -> Dict:
        """
        Exit the filled position (take directional loss).

        Args:
            opportunity: Opportunity dictionary
            filled_token: Token to sell
            size: Size to sell
            filled_side: Which side filled
            dry_run: If True, simulate

        Returns:
            Result dictionary
        """
        print(f"Exiting {filled_side} position at market...")

        # Get best bid (sell at market)
        bids, _ = self.client.get_order_book(filled_token)

        if bids.empty:
            return {'success': False, 'error': 'No bids to sell into'}

        best_bid = float(bids.iloc[-1]['price'])

        # Calculate loss
        entry_price = (opportunity['yes_ask_price'] if filled_side == 'YES'
                       else opportunity['no_ask_price'])
        loss_per_share = entry_price - best_bid
        total_loss = loss_per_share * size

        print(f"Exit price: ${best_bid:.4f}")
        print(f"Entry price: ${entry_price:.4f}")
        print(f"Loss: ${total_loss:.2f}")

        if dry_run:
            print("[DRY RUN] Exit not executed")
            return {
                'success': True,
                'strategy': 'exit_position',
                'exit_price': best_bid,
                'loss': total_loss
            }

        # Place sell order
        response = self.client.create_order(
            marketId=filled_token,
            action='SELL',
            price=best_bid,
            size=size,
            neg_risk=opportunity['neg_risk']
        )

        if response:
            print("✓ Position exited")
            # Track loss
            self.daily_pnl -= total_loss
            return {
                'success': True,
                'strategy': 'exit_position',
                'exit_price': best_bid,
                'loss': total_loss
            }
        else:
            print("❌ Exit failed")
            return {'success': False, 'error': 'Exit order failed'}

    def check_risk_limits(self) -> Tuple[bool, str]:
        """
        Check if risk limits allow trading.

        Returns:
            Tuple of (can_trade, reason)
        """
        # Check daily loss limit
        max_daily_loss = self.config.get('max_daily_loss', float('inf'))
        if abs(self.daily_pnl) > max_daily_loss:
            return False, f"Daily loss limit exceeded: ${abs(self.daily_pnl):.2f} > ${max_daily_loss:.2f}"

        # Check win rate
        if len(self.execution_history) >= 10:  # Need at least 10 trades
            wins = sum(1 for trade in self.execution_history if trade.get('realized_profit', 0) > 0)
            win_rate = wins / len(self.execution_history)
            min_win_rate = self.config.get('min_win_rate', 0.0)

            if win_rate < min_win_rate:
                return False, f"Win rate too low: {win_rate*100:.1f}% < {min_win_rate*100:.1f}%"

        return True, ""

    def record_execution(self, result: Dict):
        """
        Record execution result for tracking.

        Args:
            result: Execution result dictionary
        """
        self.execution_history.append(result)

        # Update daily P&L
        if result.get('success') and 'realized_profit' in result:
            self.daily_pnl += result['realized_profit']

    def get_stats(self) -> Dict:
        """
        Get execution statistics.

        Returns:
            Statistics dictionary
        """
        if not self.execution_history:
            return {'total_trades': 0}

        successful = [t for t in self.execution_history if t.get('success')]
        failed = [t for t in self.execution_history if not t.get('success')]

        total_profit = sum(t.get('realized_profit', 0) for t in successful)
        win_rate = len(successful) / len(self.execution_history) if self.execution_history else 0

        return {
            'total_trades': len(self.execution_history),
            'successful': len(successful),
            'failed': len(failed),
            'win_rate': win_rate,
            'total_profit': total_profit,
            'daily_pnl': self.daily_pnl,
            'avg_profit': total_profit / len(successful) if successful else 0,
        }
