"""
Market Scanner for Near-Sure Trading

Scans all Polymarket markets and filters for near-certain outcomes based on:
- Price proximity to certainty (close to 0 or 1)
- Time until market closes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings("ignore")


class NearSureMarketScanner:
    """
    Scans and filters markets for near-certain trading opportunities.
    """

    def __init__(self, client):
        """
        Initialize the market scanner.

        Args:
            client: PolymarketClient instance
        """
        self.client = client

    def get_all_markets(self) -> pd.DataFrame:
        """
        Fetch all available Polymarket markets.

        Returns:
            DataFrame with all markets
        """
        cursor = ""
        all_markets = []
        page = 0

        while True:
            try:
                markets = self.client.client.get_sampling_markets(next_cursor=cursor)

                # Validate response structure
                if not isinstance(markets, dict) or 'data' not in markets:
                    print(f"Invalid response structure on page {page + 1}")
                    break

                markets_df = pd.DataFrame(markets['data'])

                if not markets_df.empty:
                    all_markets.append(markets_df)
                    page += 1
                    print(f"Fetched page {page}: {len(markets_df)} markets (Total: {sum(len(m) for m in all_markets)})")

                # Check for next cursor
                cursor = markets.get('next_cursor')
                if cursor is None:
                    print(f"Pagination complete: {page} pages fetched")
                    break

            except Exception as e:
                # Handle end-of-pagination errors gracefully
                error_str = str(e)
                if ('400' in error_str or 'next item should be greater than or equal to 0' in error_str):
                    print(f"Pagination complete: {page} pages fetched (reached end of available markets)")
                    break
                else:
                    print(f"Error fetching markets on page {page + 1}: {e}")
                    break

        if not all_markets:
            return pd.DataFrame()

        all_df = pd.concat(all_markets, ignore_index=True)
        return all_df

    def enrich_market_data(self, row) -> Optional[Dict]:
        """
        Enrich market data with order book information.

        Args:
            row: Market row from DataFrame

        Returns:
            Dictionary with enriched market data or None on error
        """
        try:
            ret = {}
            ret['question'] = row['question']
            ret['neg_risk'] = row['neg_risk']
            ret['market_slug'] = row['market_slug']
            ret['condition_id'] = row['condition_id']
            ret['end_date_iso'] = row['end_date_iso']

            # Parse closing time - handle None and timezone issues
            if row['end_date_iso'] is None:
                return None

            ret['end_date'] = pd.to_datetime(row['end_date_iso'])

            # Convert to timezone-naive UTC for comparison
            if ret['end_date'].tzinfo is not None:
                ret['end_date'] = ret['end_date'].tz_convert('UTC').tz_localize(None)

            # Use timezone-naive datetime for comparison
            now = datetime.utcnow()
            ret['hours_until_close'] = (ret['end_date'] - now).total_seconds() / 3600

            # Get token information
            if 'tokens' not in row or len(row['tokens']) < 2:
                return None

            ret['token1'] = row['tokens'][0]['token_id']
            ret['token2'] = row['tokens'][1]['token_id']
            ret['answer1'] = row['tokens'][0]['outcome']
            ret['answer2'] = row['tokens'][1]['outcome']

            # Get order book for token1
            bids_df, asks_df = self.client.get_order_book(ret['token1'])

            if bids_df.empty or asks_df.empty:
                return None

            ret['best_bid'] = float(bids_df.iloc[-1]['price'])
            ret['best_ask'] = float(asks_df.iloc[-1]['price'])
            ret['midpoint'] = (ret['best_bid'] + ret['best_ask']) / 2
            ret['spread'] = ret['best_ask'] - ret['best_bid']

            # Get market parameters
            ret['tick_size'] = row['minimum_tick_size']

            if 'rewards' in row:
                ret['min_size'] = row['rewards'].get('min_size', 10)
                ret['max_spread'] = row['rewards'].get('max_spread', 5)
            else:
                ret['min_size'] = 10
                ret['max_spread'] = 5

            # Calculate certainty score (0 = uncertain, 1 = very certain)
            # Markets near 0 or 1 are more certain
            distance_from_extreme = min(ret['midpoint'], 1 - ret['midpoint'])
            ret['certainty_score'] = 1 - (distance_from_extreme / 0.5)

            return ret

        except Exception as e:
            print(f"Error enriching market data: {e}")
            return None

    def filter_near_sure_markets(
        self,
        min_certainty: float = 0.70,
        max_hours_until_close: Optional[float] = None,
        min_hours_until_close: float = 1.0,
        min_midpoint: float = 0.85,
        max_midpoint: float = 0.15
    ) -> pd.DataFrame:
        """
        Filter markets for near-sure trading opportunities.

        Args:
            min_certainty: Minimum certainty score (0-1)
            max_hours_until_close: Maximum hours until market closes (None = no limit)
            min_hours_until_close: Minimum hours until close (avoid markets closing too soon)
            min_midpoint: Minimum price to consider (for bullish near-sure)
            max_midpoint: Maximum price to consider (for bearish near-sure)

        Returns:
            DataFrame of filtered markets sorted by closing time
        """
        print("Fetching all markets...")
        all_markets = self.get_all_markets()

        if all_markets.empty:
            print("No markets found")
            return pd.DataFrame()

        print(f"Found {len(all_markets)} total markets")
        print("Enriching market data...")

        enriched_markets = []
        for idx, row in all_markets.iterrows():
            enriched = self.enrich_market_data(row)
            if enriched:
                enriched_markets.append(enriched)

            # Progress indicator
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(all_markets)} markets...")

        if not enriched_markets:
            print("No markets could be enriched")
            return pd.DataFrame()

        df = pd.DataFrame(enriched_markets)

        print(f"Successfully enriched {len(df)} markets")
        print("Applying filters...")

        # Filter by price certainty (high or low)
        price_filter = (df['midpoint'] >= min_midpoint) | (df['midpoint'] <= max_midpoint)

        # Filter by certainty score
        certainty_filter = df['certainty_score'] >= min_certainty

        # Filter by time until close
        time_filter = df['hours_until_close'] >= min_hours_until_close
        if max_hours_until_close:
            time_filter = time_filter & (df['hours_until_close'] <= max_hours_until_close)

        # Combine filters
        filtered_df = df[price_filter & certainty_filter & time_filter].copy()

        # Sort by closing time (soonest first)
        filtered_df = filtered_df.sort_values('hours_until_close')

        print(f"Found {len(filtered_df)} near-sure markets")

        return filtered_df

    def format_market_display(self, market: Dict) -> str:
        """
        Format market information for display.

        Args:
            market: Market dictionary

        Returns:
            Formatted string for display
        """
        question = market['question'][:55] + "..." if len(market['question']) > 55 else market['question']

        close_time = market['end_date'].strftime('%m/%d %H:%M')
        hours = market['hours_until_close']

        # Determine which side is near-sure
        if market['midpoint'] >= 0.85:
            side = "YES"
            price = market['midpoint']
        else:
            side = "NO"
            price = 1 - market['midpoint']

        return (
            f"{question:60s} | "
            f"{side:3s} @ {price:.3f} | "
            f"Close: {close_time} ({hours:.1f}h) | "
            f"Spread: {market['spread']:.3f}"
        )
