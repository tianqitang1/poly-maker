"""
Risk Manager for Near-Sure Trading

Monitors all positions in the account and implements stop-loss protection.
Since the account is dedicated to near-sure trades, all positions are monitored.
"""

import pandas as pd
import time
from typing import Dict, List, Optional
from datetime import datetime


class NearSureRiskManager:
    """
    Manages risk for near-sure trading account.
    Monitors all positions and implements stop-loss protection.
    """

    def __init__(self, client, stop_loss_pct: float = -0.10, logger=None):
        """
        Initialize the risk manager.

        Args:
            client: PolymarketClient instance
            stop_loss_pct: Stop loss percentage (e.g., -0.10 for -10%)
            logger: BotLogger instance (optional)
        """
        self.client = client
        self.stop_loss_pct = stop_loss_pct
        self.positions_cache = {}
        self.logger = logger

    def get_all_positions(self) -> pd.DataFrame:
        """
        Get all positions for the account.

        Returns:
            DataFrame of all positions
        """
        try:
            positions_df = self.client.get_all_positions()

            if positions_df.empty:
                return pd.DataFrame()

            # Filter out very small positions (dust)
            positions_df = positions_df[positions_df['size'] >= 1].copy()

            return positions_df

        except Exception as e:
            print(f"Error fetching positions: {e}")
            return pd.DataFrame()

    def calculate_position_pnl(self, position: Dict) -> Dict:
        """
        Calculate PnL for a position.

        Args:
            position: Position dictionary with size, avgPrice, market data

        Returns:
            Dictionary with PnL metrics
        """
        try:
            size = float(position['size'])
            avg_price = float(position['avgPrice'])
            token_id = position['asset']

            # Get current market price
            bids_df, asks_df = self.client.get_order_book(token_id)

            if bids_df.empty or asks_df.empty:
                return {
                    'valid': False,
                    'error': 'Empty order book'
                }

            # Use best bid as current exit price (conservative)
            current_price = float(bids_df.iloc[-1]['price'])

            # Calculate PnL
            unrealized_pnl = (current_price - avg_price) * size
            pnl_pct = (current_price - avg_price) / avg_price if avg_price > 0 else 0

            return {
                'valid': True,
                'token_id': token_id,
                'size': size,
                'avg_price': avg_price,
                'current_price': current_price,
                'unrealized_pnl': unrealized_pnl,
                'pnl_pct': pnl_pct,
                'market': position.get('market', 'Unknown'),
            }

        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }

    def check_stop_loss_trigger(self, position_pnl: Dict) -> bool:
        """
        Check if position should trigger stop loss.

        Args:
            position_pnl: Position PnL dictionary

        Returns:
            True if stop loss should trigger
        """
        if not position_pnl.get('valid', False):
            return False

        pnl_pct = position_pnl['pnl_pct']

        return pnl_pct <= self.stop_loss_pct

    def execute_stop_loss(self, position_pnl: Dict, dry_run: bool = False) -> bool:
        """
        Execute stop loss by selling position at market.

        Args:
            position_pnl: Position PnL dictionary
            dry_run: If True, simulate without executing

        Returns:
            True if successful
        """
        try:
            token_id = position_pnl['token_id']
            size = position_pnl['size']
            current_price = position_pnl['current_price']

            if self.logger:
                self.logger.warning(f"STOP LOSS TRIGGERED - Market: {position_pnl['market']}, "
                                   f"PnL: ${position_pnl['unrealized_pnl']:.2f} ({position_pnl['pnl_pct']*100:.2f}%)")
            else:
                print(f"\n{'!'*80}")
                print(f"STOP LOSS TRIGGERED")
                print(f"{'!'*80}")
                print(f"Market: {position_pnl['market']}")
                print(f"Token ID: {token_id}")
                print(f"Size: {size:.2f}")
                print(f"Entry Price: {position_pnl['avg_price']:.4f}")
                print(f"Current Price: {current_price:.4f}")
                print(f"PnL: ${position_pnl['unrealized_pnl']:.2f} ({position_pnl['pnl_pct']*100:.2f}%)")
                print(f"Stop Loss Threshold: {self.stop_loss_pct*100:.2f}%")
                print(f"{'!'*80}")

            if dry_run:
                print("[DRY RUN] Stop loss not executed")
                return True

            # Cancel any existing orders for this token first
            if self.logger:
                self.logger.info(f"Cancelling existing orders for token {token_id}")
            else:
                print(f"Cancelling existing orders for token {token_id}...")

            self.client.cancel_all_asset(token_id)
            time.sleep(0.5)

            # Sell at best bid (market order)
            # We use best bid price to ensure immediate execution
            if self.logger:
                self.logger.info(f"Placing market SELL order at {current_price:.4f}")
            else:
                print(f"Placing market sell order at {current_price:.4f}...")

            # Note: We need to determine if it's a neg_risk market
            # For simplicity, try regular first, then neg_risk if it fails
            neg_risk = False
            if 'neg_risk' in position_pnl:
                neg_risk = position_pnl['neg_risk']

            response = self.client.create_order(
                marketId=token_id,
                action='SELL',
                price=current_price,
                size=size,
                neg_risk=neg_risk
            )

            if response:
                print("✓ Stop loss executed successfully!")

                # Log stop loss
                if self.logger:
                    self.logger.log_stop_loss(
                        market=position_pnl['market'],
                        position=size,
                        entry_price=position_pnl['avg_price'],
                        exit_price=current_price,
                        pnl=position_pnl['unrealized_pnl'],
                        reason=f"PnL: {position_pnl['pnl_pct']*100:.2f}% below threshold {self.stop_loss_pct*100:.2f}%"
                    )

                return True
            else:
                print("❌ Stop loss execution failed")
                if self.logger:
                    self.logger.error(f"Stop loss execution failed for {position_pnl['market']}")
                return False

        except Exception as e:
            print(f"❌ Error executing stop loss: {e}")
            import traceback
            traceback.print_exc()
            return False

    def monitor_positions(self, dry_run: bool = False) -> Dict:
        """
        Monitor all positions and trigger stop losses if needed.

        Args:
            dry_run: If True, simulate without executing

        Returns:
            Dictionary with monitoring results
        """
        print(f"\n{'='*80}")
        print(f"Risk Monitoring - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Stop Loss Threshold: {self.stop_loss_pct*100:.2f}%")
        print(f"{'='*80}")

        positions_df = self.get_all_positions()

        if positions_df.empty:
            print("No positions to monitor")
            return {
                'total_positions': 0,
                'stop_losses_triggered': 0,
                'positions_monitored': []
            }

        results = {
            'total_positions': len(positions_df),
            'stop_losses_triggered': 0,
            'positions_monitored': []
        }

        print(f"\nMonitoring {len(positions_df)} position(s)...\n")

        for idx, position in positions_df.iterrows():
            pnl = self.calculate_position_pnl(position)

            if not pnl.get('valid', False):
                print(f"⚠️  {position.get('market', 'Unknown')}: Could not calculate PnL - {pnl.get('error', 'Unknown error')}")
                continue

            # Display position status
            status = "🔴" if self.check_stop_loss_trigger(pnl) else "🟢"
            print(f"{status} {pnl['market'][:50]:50s} | "
                  f"PnL: ${pnl['unrealized_pnl']:7.2f} ({pnl['pnl_pct']*100:6.2f}%) | "
                  f"Size: {pnl['size']:7.2f} @ {pnl['avg_price']:.4f}")

            results['positions_monitored'].append(pnl)

            # Check for stop loss
            if self.check_stop_loss_trigger(pnl):
                success = self.execute_stop_loss(pnl, dry_run=dry_run)
                if success:
                    results['stop_losses_triggered'] += 1

                # Brief delay after stop loss execution
                if not dry_run:
                    time.sleep(2)

        print(f"\n{'='*80}")
        print(f"Stop Losses Triggered: {results['stop_losses_triggered']}")
        print(f"{'='*80}\n")

        return results

    def continuous_monitoring(
        self,
        check_interval: int = 30,
        dry_run: bool = False
    ):
        """
        Continuously monitor positions at regular intervals.

        Args:
            check_interval: Seconds between checks
            dry_run: If True, simulate without executing

        Note: Run this in a separate thread/process or as main loop
        """
        print(f"\n{'='*80}")
        print("Starting Continuous Risk Monitoring")
        print(f"Check Interval: {check_interval}s")
        print(f"Stop Loss: {self.stop_loss_pct*100:.2f}%")
        print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
        print(f"{'='*80}\n")
        print("Press Ctrl+C to stop\n")

        try:
            while True:
                self.monitor_positions(dry_run=dry_run)

                print(f"Next check in {check_interval}s...\n")
                time.sleep(check_interval)

        except KeyboardInterrupt:
            print("\n\nStopping risk monitoring...")
            print("Final position check...")
            self.monitor_positions(dry_run=dry_run)
            print("Risk monitoring stopped.")
