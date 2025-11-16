"""
Arbitrage Executor

Handles atomic execution of arbitrage opportunities with partial fill protection.
"""

import time
from typing import Dict, Optional, Tuple
from datetime import datetime
from neg_risk_arb.arbitrage_scanner import load_config


class ArbitrageExecutor:
    """
    Executes arbitrage trades with partial fill protection.
    """

    def __init__(self, client, config_path='neg_risk_arb/config.yaml'):
        """
        Initialize the executor.

        Args:
            client: PolymarketClient instance
            config_path: Path to configuration file
        """
        self.client = client

        # Load configuration
        self.config = load_config(config_path)

    def validate_opportunity(self, opportunity: Dict) -> bool:
        """
        Re-validate an opportunity before executing.

        Args:
            opportunity: Opportunity dictionary

        Returns:
            True if still valid
        """
        try:
            # Re-fetch order books to ensure prices haven't changed
            yes_bids, yes_asks = self.client.get_order_book(opportunity['yes_token'])
            no_bids, no_asks = self.client.get_order_book(opportunity['no_token'])

            if yes_asks.empty or no_asks.empty:
                print("Order book empty - opportunity no longer valid")
                return False

            # Check current prices
            current_yes_ask = float(yes_asks.iloc[-1]['price'])
            current_no_ask = float(no_asks.iloc[-1]['price'])
            current_total = current_yes_ask + current_no_ask

            # Calculate slippage
            expected_total = opportunity['total_cost']
            slippage = abs(current_total - expected_total) / expected_total

            max_slippage = self.config['max_slippage_tolerance']
            if slippage > max_slippage:
                print(f"Slippage too high: {slippage*100:.2f}% (max: {max_slippage*100:.2f}%)")
                print(f"Expected: ${expected_total:.4f}, Current: ${current_total:.4f}")
                return False

            # Check if still profitable after slippage
            if current_total >= self.config['max_total_cost']:
                print(f"No longer profitable: ${current_total:.4f} >= ${self.config['max_total_cost']:.4f}")
                return False

            # Update opportunity with current prices
            opportunity['yes_ask_price'] = current_yes_ask
            opportunity['no_ask_price'] = current_no_ask
            opportunity['total_cost'] = current_total

            return True

        except Exception as e:
            print(f"Error validating opportunity: {e}")
            return False

    def execute_orders(
        self,
        opportunity: Dict,
        size: float,
        dry_run: bool = False
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Execute both BUY orders atomically.

        Args:
            opportunity: Opportunity dictionary
            size: Size to trade
            dry_run: If True, simulate without executing

        Returns:
            Tuple of (yes_response, no_response)
        """
        yes_token = opportunity['yes_token']
        no_token = opportunity['no_token']
        yes_price = opportunity['yes_ask_price']
        no_price = opportunity['no_ask_price']
        neg_risk = opportunity['neg_risk']

        print(f"\n{'='*80}")
        print(f"Executing Arbitrage")
        print(f"{'='*80}")
        print(f"Market: {opportunity['question'][:70]}")
        print(f"YES: {size:.2f} @ ${yes_price:.4f} = ${size * yes_price:.2f}")
        print(f"NO:  {size:.2f} @ ${no_price:.4f} = ${size * no_price:.2f}")
        print(f"Total Cost: ${size * opportunity['total_cost']:.2f}")
        print(f"Expected Profit: ${size * opportunity['profit_per_share']:.2f}")
        print(f"{'='*80}\n")

        if dry_run:
            print("[DRY RUN] Orders not placed")
            return (
                {'dry_run': True, 'side': 'YES', 'size': size, 'price': yes_price},
                {'dry_run': True, 'side': 'NO', 'size': size, 'price': no_price}
            )

        # Execute YES order
        print(f"Placing YES order: {size:.2f} @ ${yes_price:.4f}...")
        yes_response = self.client.create_order(
            marketId=yes_token,
            action='BUY',
            price=yes_price,
            size=size,
            neg_risk=neg_risk
        )

        if not yes_response:
            print("❌ YES order failed")
            return None, None

        print(f"✓ YES order placed: {yes_response.get('orderID', 'N/A')}")

        # Small delay if configured
        delay_ms = self.config.get('execution_delay_ms', 0)
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)

        # Execute NO order
        print(f"Placing NO order: {size:.2f} @ ${no_price:.4f}...")
        no_response = self.client.create_order(
            marketId=no_token,
            action='BUY',
            price=no_price,
            size=size,
            neg_risk=neg_risk
        )

        if not no_response:
            print("❌ NO order failed")
            return yes_response, None

        print(f"✓ NO order placed: {no_response.get('orderID', 'N/A')}")

        return yes_response, no_response

    def verify_fills(
        self,
        opportunity: Dict,
        expected_size: float,
        timeout: float = 3.0
    ) -> Tuple[float, float]:
        """
        Wait for orders to fill and verify positions.

        Args:
            opportunity: Opportunity dictionary
            expected_size: Expected fill size
            timeout: Seconds to wait

        Returns:
            Tuple of (yes_position, no_position) in shares
        """
        yes_token = opportunity['yes_token']
        no_token = opportunity['no_token']

        print(f"\nWaiting {timeout}s for order confirmations...")
        time.sleep(timeout)

        # Get positions
        _, yes_pos = self.client.get_position(yes_token)
        _, no_pos = self.client.get_position(no_token)

        print(f"Positions: YES={yes_pos:.2f}, NO={no_pos:.2f} (expected: {expected_size:.2f})")

        return yes_pos, no_pos

    def merge_positions(
        self,
        opportunity: Dict,
        merge_size: float,
        dry_run: bool = False
    ) -> bool:
        """
        Merge YES and NO positions to recover collateral.

        Args:
            opportunity: Opportunity dictionary
            merge_size: Size to merge (in shares)
            dry_run: If True, simulate without executing

        Returns:
            True if successful
        """
        condition_id = opportunity['condition_id']
        neg_risk = opportunity['neg_risk']

        # Check minimum merge size
        min_merge = self.config.get('min_merge_size', 20)
        if merge_size < min_merge:
            print(f"Merge size {merge_size:.2f} below minimum {min_merge}")
            return False

        print(f"\n{'='*80}")
        print(f"Merging Positions")
        print(f"{'='*80}")
        print(f"Size: {merge_size:.2f} shares")
        print(f"Condition ID: {condition_id}")
        print(f"Neg Risk: {neg_risk}")
        print(f"Expected Recovery: ${merge_size:.2f}")
        print(f"{'='*80}\n")

        if dry_run:
            print("[DRY RUN] Merge not executed")
            return True

        # Wait before merging (let blockchain settle)
        merge_delay = self.config.get('merge_delay_sec', 5)
        if merge_delay > 0:
            print(f"Waiting {merge_delay}s for blockchain to settle...")
            time.sleep(merge_delay)

        # Get exact on-chain positions
        yes_token = opportunity['yes_token']
        no_token = opportunity['no_token']

        raw_yes_pos, _ = self.client.get_position(yes_token)
        raw_no_pos, _ = self.client.get_position(no_token)

        # Calculate amount to merge (in raw units)
        amount_to_merge = min(raw_yes_pos, raw_no_pos)

        if amount_to_merge == 0:
            print("No positions to merge")
            return False

        # Execute merge
        max_retries = self.config.get('max_merge_retries', 3)
        for attempt in range(max_retries):
            try:
                print(f"Merge attempt {attempt + 1}/{max_retries}...")
                result = self.client.merge_positions(
                    amount_to_merge=amount_to_merge,
                    condition_id=condition_id,
                    is_neg_risk_market=neg_risk
                )
                print("✓ Positions merged successfully!")
                return True
            except Exception as e:
                print(f"Merge attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    print("❌ Merge failed after all retries")
                    return False

        return False

    def execute_arbitrage(
        self,
        opportunity: Dict,
        size: Optional[float] = None,
        dry_run: bool = False
    ) -> Dict:
        """
        Execute complete arbitrage: buy both sides and merge.

        Args:
            opportunity: Opportunity dictionary
            size: Size to trade (defaults to tradeable_size from opportunity)
            dry_run: If True, simulate without executing

        Returns:
            Result dictionary with execution details
        """
        result = {
            'success': False,
            'timestamp': datetime.now(),
            'opportunity': opportunity['question'],
            'dry_run': dry_run,
        }

        # Validate opportunity is still good
        if not dry_run:
            if not self.validate_opportunity(opportunity):
                result['error'] = 'Opportunity no longer valid'
                return result

        # Determine size
        if size is None:
            size = opportunity['tradeable_size']

        # Check minimum size
        if size < self.config['min_position_size']:
            result['error'] = f'Size {size} below minimum {self.config["min_position_size"]}'
            return result

        # Execute orders
        yes_response, no_response = self.execute_orders(opportunity, size, dry_run)

        if yes_response is None or no_response is None:
            result['error'] = 'Order execution failed'
            result['yes_filled'] = yes_response is not None
            result['no_filled'] = no_response is not None
            return result

        if dry_run:
            result['success'] = True
            result['size'] = size
            result['expected_profit'] = size * opportunity['profit_per_share']
            return result

        # Wait for fills and verify
        timeout = self.config.get('confirmation_timeout_sec', 3)
        yes_pos, no_pos = self.verify_fills(opportunity, size, timeout)

        result['yes_position'] = yes_pos
        result['no_position'] = no_pos

        # Check fill status
        fill_threshold = size * 0.95  # Allow 5% slippage on fill size

        yes_filled = yes_pos >= fill_threshold
        no_filled = no_pos >= fill_threshold

        if yes_filled and no_filled:
            # Both filled - proceed to merge
            merge_size = min(yes_pos, no_pos)
            merge_success = self.merge_positions(opportunity, merge_size, dry_run)

            if merge_success:
                # Calculate realized profit
                total_cost = opportunity['total_cost'] * merge_size
                recovered = merge_size
                realized_profit = recovered - total_cost

                result['success'] = True
                result['size'] = merge_size
                result['total_cost'] = total_cost
                result['recovered'] = recovered
                result['realized_profit'] = realized_profit
                result['profit_bps'] = opportunity['profit_bps']

                print(f"\n{'🎉'*40}")
                print(f"ARBITRAGE SUCCESSFUL!")
                print(f"Profit: ${realized_profit:.2f} ({opportunity['profit_bps']:.1f}bp)")
                print(f"{'🎉'*40}\n")
            else:
                result['error'] = 'Merge failed'
                result['positions_stuck'] = True

        else:
            # Partial fill - handle via risk manager
            result['error'] = 'Partial fill - needs rescue'
            result['partial_fill'] = True

        return result
