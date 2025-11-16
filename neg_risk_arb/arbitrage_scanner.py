"""
Arbitrage Scanner for Negative Risk Markets

Scans Polymarket for opportunities where YES ask + NO ask < $1.00,
allowing for risk-free arbitrage through position merging.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import yaml
import warnings
warnings.filterwarnings("ignore")


class ArbitrageScanner:
    """
    Scans for negative risk arbitrage opportunities.
    """

    def __init__(self, client, config_path='neg_risk_arb/config.yaml'):
        """
        Initialize the arbitrage scanner.

        Args:
            client: PolymarketClient instance
            config_path: Path to configuration file
        """
        self.client = client

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get_all_markets(self) -> pd.DataFrame:
        """
        Fetch all available Polymarket markets.

        Returns:
            DataFrame with all markets
        """
        cursor = ""
        all_markets = []

        while True:
            try:
                markets = self.client.client.get_sampling_markets(next_cursor=cursor)
                markets_df = pd.DataFrame(markets['data'])
                cursor = markets['next_cursor']
                all_markets.append(markets_df)

                if cursor is None:
                    break
            except Exception as e:
                print(f"Error fetching markets: {e}")
                break

        if not all_markets:
            return pd.DataFrame()

        all_df = pd.concat(all_markets)
        all_df = all_df.reset_index(drop=True)
        return all_df

    def calculate_arbitrage_opportunity(self, market_row) -> Optional[Dict]:
        """
        Calculate arbitrage opportunity for a single market.

        Args:
            market_row: Market row from DataFrame

        Returns:
            Dictionary with opportunity details or None if no opportunity
        """
        try:
            # Only consider negative risk markets
            if self.config['require_neg_risk'] and not market_row['neg_risk']:
                return None

            # Check if market has required fields
            if 'tokens' not in market_row or len(market_row['tokens']) < 2:
                return None

            # Get token information
            yes_token = market_row['tokens'][0]['token_id']
            no_token = market_row['tokens'][1]['token_id']
            yes_outcome = market_row['tokens'][0]['outcome']
            no_outcome = market_row['tokens'][1]['outcome']

            # Get order books
            yes_bids, yes_asks = self.client.get_order_book(yes_token)
            no_bids, no_asks = self.client.get_order_book(no_token)

            if yes_asks.empty or no_asks.empty:
                return None

            # Get best ask prices and sizes
            yes_ask_price = float(yes_asks.iloc[-1]['price'])
            yes_ask_size = float(yes_asks.iloc[-1]['size'])
            no_ask_price = float(no_asks.iloc[-1]['price'])
            no_ask_size = float(no_asks.iloc[-1]['size'])

            # Calculate total cost
            total_cost = yes_ask_price + no_ask_price

            # Check if this is an arbitrage opportunity
            max_cost = self.config['max_total_cost']
            if total_cost >= max_cost:
                return None

            # Calculate profit
            profit_per_share = 1.0 - total_cost
            profit_bps = profit_per_share * 10000  # basis points

            # Check minimum profit threshold
            if profit_bps < self.config['min_profit_bps']:
                return None

            # Check liquidity requirements
            min_liquidity = self.config['min_liquidity_per_side']
            if yes_ask_size < min_liquidity or no_ask_size < min_liquidity:
                if self.config['require_both_sides_liquid']:
                    return None

            # Available liquidity (how much we can trade)
            available_liquidity = min(yes_ask_size, no_ask_size)

            # Cap at max position size
            max_size = self.config['max_position_size']
            tradeable_size = min(available_liquidity, max_size)

            # Check minimum position size
            if tradeable_size < self.config['min_position_size']:
                return None

            # Parse closing time
            end_date = pd.to_datetime(market_row['end_date_iso'])
            hours_until_close = (end_date - datetime.now()).total_seconds() / 3600

            # Filter by closing time
            min_hours = self.config['min_hours_until_close']
            max_hours = self.config['max_hours_until_close']

            if hours_until_close < min_hours:
                return None
            if max_hours and hours_until_close > max_hours:
                return None

            # Check for excluded keywords
            question = market_row['question'].lower()
            for keyword in self.config.get('exclude_keywords', []):
                if keyword.lower() in question:
                    return None

            # Calculate expected profit
            expected_profit = profit_per_share * tradeable_size

            # Build opportunity dictionary
            opportunity = {
                'question': market_row['question'],
                'market_slug': market_row.get('market_slug', ''),
                'condition_id': market_row['condition_id'],
                'yes_token': yes_token,
                'no_token': no_token,
                'yes_outcome': yes_outcome,
                'no_outcome': no_outcome,
                'yes_ask_price': yes_ask_price,
                'no_ask_price': no_ask_price,
                'total_cost': total_cost,
                'profit_per_share': profit_per_share,
                'profit_bps': profit_bps,
                'yes_ask_size': yes_ask_size,
                'no_ask_size': no_ask_size,
                'available_liquidity': available_liquidity,
                'tradeable_size': tradeable_size,
                'expected_profit': expected_profit,
                'expected_cost': total_cost * tradeable_size,
                'end_date': end_date,
                'hours_until_close': hours_until_close,
                'neg_risk': market_row['neg_risk'],
            }

            return opportunity

        except Exception as e:
            # Silently skip markets with errors
            return None

    def scan_for_opportunities(self, verbose=True) -> pd.DataFrame:
        """
        Scan all markets for arbitrage opportunities.

        Args:
            verbose: If True, print progress

        Returns:
            DataFrame of opportunities sorted by profit
        """
        if verbose:
            print("Fetching all markets...")

        all_markets = self.get_all_markets()

        if all_markets.empty:
            if verbose:
                print("No markets found")
            return pd.DataFrame()

        if verbose:
            print(f"Found {len(all_markets)} total markets")
            print("Scanning for arbitrage opportunities...")

        opportunities = []

        for idx, row in all_markets.iterrows():
            opp = self.calculate_arbitrage_opportunity(row)
            if opp:
                opportunities.append(opp)

            # Progress indicator
            if verbose and (idx + 1) % 50 == 0:
                print(f"Scanned {idx + 1}/{len(all_markets)} markets... "
                      f"Found {len(opportunities)} opportunities")

        if not opportunities:
            if verbose:
                print("No arbitrage opportunities found")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(opportunities)

        # Sort by configured preference
        sort_by = self.config.get('sort_opportunities_by', 'profit')
        if sort_by == 'profit':
            df = df.sort_values('profit_bps', ascending=False)
        elif sort_by == 'liquidity':
            df = df.sort_values('available_liquidity', ascending=False)
        elif sort_by == 'closing_time':
            df = df.sort_values('hours_until_close')

        if verbose:
            print(f"\n✓ Found {len(df)} arbitrage opportunities!")

        return df

    def format_opportunity(self, opp: Dict) -> str:
        """
        Format an opportunity for display.

        Args:
            opp: Opportunity dictionary

        Returns:
            Formatted string
        """
        question = opp['question'][:60] + "..." if len(opp['question']) > 60 else opp['question']

        return (
            f"{question:65s} | "
            f"Profit: {opp['profit_bps']:5.1f}bp (${opp['expected_profit']:.2f}) | "
            f"Cost: ${opp['total_cost']:.4f} | "
            f"Size: ${opp['tradeable_size']:.0f} | "
            f"Close: {opp['hours_until_close']:.1f}h"
        )

    def display_opportunities(self, opportunities_df: pd.DataFrame, max_display=None):
        """
        Display opportunities in a formatted table.

        Args:
            opportunities_df: DataFrame of opportunities
            max_display: Maximum number to display (None = all)
        """
        if opportunities_df.empty:
            print("\nNo arbitrage opportunities available.")
            return

        if max_display is None:
            max_display = self.config.get('max_opportunities_to_display', 10)

        display_df = opportunities_df.head(max_display)

        print(f"\n{'='*120}")
        print(f"ARBITRAGE OPPORTUNITIES ({len(opportunities_df)} found, showing top {len(display_df)})")
        print(f"{'='*120}\n")

        for idx, row in display_df.iterrows():
            print(f"{len(display_df) - idx}. {self.format_opportunity(row.to_dict())}")
            if self.config.get('show_order_book_depth', False):
                print(f"   YES: {row['yes_ask_size']:.0f} @ ${row['yes_ask_price']:.4f} | "
                      f"NO: {row['no_ask_size']:.0f} @ ${row['no_ask_price']:.4f}")

        print(f"\n{'='*120}")

        # Summary statistics
        total_profit = display_df['expected_profit'].sum()
        avg_profit_bps = display_df['profit_bps'].mean()
        total_capital = display_df['expected_cost'].sum()

        print(f"Summary: Avg Profit: {avg_profit_bps:.1f}bp | "
              f"Total Potential Profit: ${total_profit:.2f} | "
              f"Total Capital Required: ${total_capital:.2f}")
        print(f"{'='*120}\n")
