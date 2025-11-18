"""
Order Manager for Near-Sure Trading

Handles passive order placement at existing bid prices near the midpoint,
avoiding aggressive market orders.
"""

import pandas as pd
from typing import Dict, Tuple, Optional
import time


class NearSureOrderManager:
    """
    Manages order placement for near-sure trading strategies.
    Places passive limit orders at existing bid prices near the midpoint.
    """

    def __init__(self, client, logger=None):
        """
        Initialize the order manager.

        Args:
            client: PolymarketClient instance
            logger: BotLogger instance (optional)
        """
        self.client = client
        self.logger = logger

    def find_passive_bid_price(
        self,
        token_id: str,
        target_side: str,
        min_size_threshold: float = 10.0
    ) -> Optional[Tuple[float, float]]:
        """
        Find the best passive bid price near the midpoint.

        Args:
            token_id: Token ID to trade
            target_side: 'YES' or 'NO' - which side to buy
            min_size_threshold: Minimum size at price level to consider

        Returns:
            Tuple of (price, size) or None if no suitable price found
        """
        try:
            bids_df, asks_df = self.client.get_order_book(token_id)
        except (ValueError, TypeError) as e:
            print(f"Error getting order book for {token_id}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error getting order book for {token_id}: {e}")
            return None

        try:
            if bids_df.empty or asks_df.empty:
                print(f"Empty order book for token {token_id}")
                return None

            # Get best bid and ask
            best_bid = float(bids_df.iloc[-1]['price'])
            best_ask = float(asks_df.iloc[-1]['price'])
            midpoint = (best_bid + best_ask) / 2

            # Filter bids with sufficient size
            bids_with_size = bids_df[bids_df['size'] >= min_size_threshold]

            if bids_with_size.empty:
                print(f"No bids with size >= {min_size_threshold}")
                # Fallback to best bid
                return (best_bid, float(bids_df.iloc[-1]['size']))

            # Find the closest bid to midpoint (but not exceeding it)
            valid_bids = bids_with_size[bids_with_size['price'] <= midpoint]

            if valid_bids.empty:
                # No bids below midpoint, use best bid
                return (best_bid, float(bids_df.iloc[-1]['size']))

            # Get the highest bid that's still below midpoint
            closest_bid = valid_bids.iloc[-1]
            price = float(closest_bid['price'])
            size = float(closest_bid['size'])

            print(f"Found passive bid: {price:.4f} (midpoint: {midpoint:.4f}, size at level: {size:.2f})")

            return (price, size)

        except Exception as e:
            print(f"Error finding passive bid price: {e}")
            return None

    def calculate_order_size(
        self,
        target_amount: float,
        price: float,
        min_size: float
    ) -> float:
        """
        Calculate appropriate order size based on target amount and price.

        Args:
            target_amount: Target trade amount in USDC
            price: Order price
            min_size: Minimum order size for the market

        Returns:
            Calculated order size
        """
        # For prediction markets, size is in USDC terms
        size = target_amount

        # Ensure minimum size compliance
        if size < min_size:
            print(f"Warning: Calculated size {size:.2f} below minimum {min_size:.2f}, adjusting to minimum")
            size = min_size

        return round(size, 2)

    def place_passive_order(
        self,
        market: Dict,
        trade_amount: float,
        dry_run: bool = False
    ) -> Optional[Dict]:
        """
        Place a passive limit order for a near-sure market.

        Args:
            market: Market dictionary with token IDs and parameters
            trade_amount: Trade amount in USDC
            dry_run: If True, simulate order without actually placing it

        Returns:
            Order response dictionary or None on error
        """
        try:
            # Determine which token to buy based on price
            if market['midpoint'] >= 0.85:
                # Buy YES token (token1)
                token_id = market['token1']
                side = 'YES'
            else:
                # Buy NO token (token2) - which means YES on the opposite outcome
                token_id = market['token2']
                side = 'NO'

            print(f"\n{'='*80}")
            print(f"Market: {market['question'][:70]}")
            print(f"Side: {side} | Midpoint: {market['midpoint']:.4f}")
            print(f"{'='*80}")

            # Find passive bid price
            price_info = self.find_passive_bid_price(
                token_id=token_id,
                target_side=side,
                min_size_threshold=market.get('min_size', 10)
            )

            if not price_info:
                print("❌ Could not find suitable bid price")
                return None

            price, level_size = price_info

            # Calculate order size
            size = self.calculate_order_size(
                target_amount=trade_amount,
                price=price,
                min_size=market.get('min_size', 10)
            )

            # Display order details
            print(f"\nOrder Details:")
            print(f"  Token ID: {token_id}")
            print(f"  Side: BUY {side}")
            print(f"  Price: {price:.4f}")
            print(f"  Size: ${size:.2f}")
            print(f"  Estimated Shares: {size/price:.2f}")
            print(f"  Neg Risk: {market['neg_risk']}")

            if dry_run:
                print("\n[DRY RUN] Order not placed")
                return {
                    'dry_run': True,
                    'token_id': token_id,
                    'side': side,
                    'price': price,
                    'size': size,
                }

            # Place the order
            if self.logger:
                self.logger.info(f"Placing BUY order for {side} @ ${price:.4f}, size: ${size:.2f}")
            else:
                print("\nPlacing order...")

            response = self.client.create_order(
                marketId=token_id,
                action='BUY',
                price=price,
                size=size,
                neg_risk=(market['neg_risk'] == 'TRUE' or market['neg_risk'] is True)
            )

            if response:
                print("✓ Order placed successfully!")
                print(f"Order ID: {response.get('orderID', 'N/A')}")

                # Log order placement
                if self.logger:
                    self.logger.log_order(
                        action='BUY',
                        market=market['question'],
                        token=token_id,
                        price=price,
                        size=size,
                        order_id=response.get('orderID', 'N/A') if isinstance(response, dict) else 'N/A',
                        side=side
                    )

                return response
            else:
                print("❌ Order placement failed (empty response)")
                if self.logger:
                    self.logger.error(f"Order placement failed for {market['question']}")
                return None

        except Exception as e:
            print(f"❌ Error placing order: {e}")
            import traceback
            traceback.print_exc()
            return None

    def place_batch_orders(
        self,
        configured_trades: list,
        dry_run: bool = False,
        delay_between_orders: float = 1.0
    ) -> Dict:
        """
        Place multiple orders with optional delay between them.

        Args:
            configured_trades: List of (market, amount) tuples
            dry_run: If True, simulate orders without placing them
            delay_between_orders: Seconds to wait between orders

        Returns:
            Dictionary with success/failure counts and results
        """
        results = {
            'successful': [],
            'failed': [],
            'total': len(configured_trades),
        }

        for idx, (market, amount) in enumerate(configured_trades, 1):
            print(f"\n{'#'*80}")
            print(f"Order {idx}/{len(configured_trades)}")
            print(f"{'#'*80}")

            response = self.place_passive_order(
                market=market,
                trade_amount=amount,
                dry_run=dry_run
            )

            if response:
                results['successful'].append({
                    'market': market['question'],
                    'amount': amount,
                    'response': response
                })
            else:
                results['failed'].append({
                    'market': market['question'],
                    'amount': amount,
                })

            # Delay between orders (except after last one)
            if idx < len(configured_trades) and delay_between_orders > 0:
                print(f"\nWaiting {delay_between_orders}s before next order...")
                time.sleep(delay_between_orders)

        # Summary
        print(f"\n{'='*80}")
        print("BATCH ORDER SUMMARY")
        print(f"{'='*80}")
        print(f"Total: {results['total']}")
        print(f"Successful: {len(results['successful'])}")
        print(f"Failed: {len(results['failed'])}")
        print(f"{'='*80}\n")

        return results
