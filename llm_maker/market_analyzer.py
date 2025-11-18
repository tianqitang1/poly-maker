"""
Market Analyzer for LLM Maker Bot

Prepares market data in a format suitable for LLM analysis.
Extracts relevant features, calculates indicators, and formats
data for the LLM to make trading decisions.
"""

import sys
import os
import pandas as pd
import time
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import poly_data.global_state as global_state
from poly_data.data_utils import get_position, get_order
from poly_data.trading_utils import get_best_bid_ask_deets


class MarketAnalyzer:
    """Analyzes markets and prepares data for LLM decision-making."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize market analyzer.

        Args:
            config: Strategy configuration dict
        """
        self.config = config
        self.market_selection_config = config.get('market_selection', {})

    def get_top_markets(self, max_markets: int = 5) -> List[str]:
        """
        Get top markets to analyze based on selection criteria.

        Args:
            max_markets: Maximum number of markets to return

        Returns:
            List of condition_ids for top markets
        """
        if not hasattr(global_state, 'df') or global_state.df is None or len(global_state.df) == 0:
            return []

        df = global_state.df.copy()

        # Apply filters
        min_volume = self.market_selection_config.get('min_volume_24h', 100)
        max_volatility = self.market_selection_config.get('max_volatility_3h', 0.15)

        # Filter by volatility
        if '3_hour' in df.columns:
            df = df[df['3_hour'] <= max_volatility]

        # TODO: Filter by volume when we have that data
        # df = df[df['volume_24h'] >= min_volume]

        # Score markets based on preferences
        df['score'] = 0.0

        # Prefer markets we already have positions in (to manage existing positions)
        for idx, row in df.iterrows():
            pos_1 = get_position(row['token1'])['size']
            pos_2 = get_position(row['token2'])['size']

            if pos_1 > 0 or pos_2 > 0:
                df.at[idx, 'score'] += 10  # High priority for existing positions

        # Prefer markets with lower volatility (easier to make markets)
        if '3_hour' in df.columns:
            # Normalize volatility score (lower is better)
            vol_score = 1 - (df['3_hour'] / df['3_hour'].max())
            df['score'] += vol_score * 5

        # Prefer markets with tighter spreads (more efficient)
        try:
            spread_scores = []
            for idx, row in df.iterrows():
                try:
                    deets = get_best_bid_ask_deets(row['condition_id'], 'token1', 20, 0.1)
                    if deets['best_bid'] and deets['best_ask']:
                        spread = deets['best_ask'] - deets['best_bid']
                        spread_score = max(0, 1 - spread / 0.10)  # Normalize to 0.10 max spread
                        spread_scores.append(spread_score * 3)
                    else:
                        spread_scores.append(0)
                except:
                    spread_scores.append(0)

            df['spread_score'] = spread_scores
            df['score'] += df['spread_score']
        except:
            pass

        # Sort by score and return top N
        df = df.sort_values('score', ascending=False)
        top_markets = df.head(max_markets)['condition_id'].tolist()

        return top_markets

    def analyze_market(self, condition_id: str) -> Optional[Dict[str, Any]]:
        """
        Analyze a single market and prepare data for LLM.

        Args:
            condition_id: Market condition ID

        Returns:
            Dict with market analysis, or None if market not found
        """
        if not hasattr(global_state, 'df') or global_state.df is None:
            return None

        # Get market row
        market_df = global_state.df[global_state.df['condition_id'] == condition_id]
        if market_df.empty:
            return None

        row = market_df.iloc[0]

        # Get order book data
        try:
            deets_token1 = get_best_bid_ask_deets(condition_id, 'token1', 100, 0.1)
            deets_token2 = get_best_bid_ask_deets(condition_id, 'token2', 100, 0.1)
        except Exception as e:
            return None

        # Calculate mid prices
        mid_price_1 = None
        mid_price_2 = None
        spread_1 = None
        spread_2 = None

        if deets_token1['best_bid'] and deets_token1['best_ask']:
            mid_price_1 = (deets_token1['best_bid'] + deets_token1['best_ask']) / 2
            spread_1 = deets_token1['best_ask'] - deets_token1['best_bid']

        if deets_token2['best_bid'] and deets_token2['best_ask']:
            mid_price_2 = (deets_token2['best_bid'] + deets_token2['best_ask']) / 2
            spread_2 = deets_token2['best_ask'] - deets_token2['best_bid']

        # Get positions
        pos_1 = get_position(row['token1'])
        pos_2 = get_position(row['token2'])

        # Calculate P&L for existing positions
        pnl_1 = None
        pnl_2 = None

        if pos_1['size'] > 0 and pos_1['avgPrice'] > 0 and mid_price_1:
            pnl_1 = ((mid_price_1 - pos_1['avgPrice']) / pos_1['avgPrice']) * 100

        if pos_2['size'] > 0 and pos_2['avgPrice'] > 0 and mid_price_2:
            pnl_2 = ((mid_price_2 - pos_2['avgPrice']) / pos_2['avgPrice']) * 100

        # Get order book imbalance (ratio of bid to ask liquidity)
        bid_liquidity_1 = deets_token1['bid_sum_within_n_percent']
        ask_liquidity_1 = deets_token1['ask_sum_within_n_percent']
        imbalance_1 = bid_liquidity_1 / ask_liquidity_1 if ask_liquidity_1 > 0 else 0

        bid_liquidity_2 = deets_token2['bid_sum_within_n_percent']
        ask_liquidity_2 = deets_token2['ask_sum_within_n_percent']
        imbalance_2 = bid_liquidity_2 / ask_liquidity_2 if ask_liquidity_2 > 0 else 0

        # Detect price trend (simple: compare current to sheet value)
        trend_1 = None
        trend_2 = None

        if mid_price_1 and 'best_bid' in row:
            sheet_price_1 = row['best_bid']
            if mid_price_1 > sheet_price_1 * 1.02:
                trend_1 = "up"
            elif mid_price_1 < sheet_price_1 * 0.98:
                trend_1 = "down"
            else:
                trend_1 = "stable"

        if mid_price_2 and 'best_ask' in row:
            sheet_price_2 = 1 - row['best_ask']
            if mid_price_2 > sheet_price_2 * 1.02:
                trend_2 = "up"
            elif mid_price_2 < sheet_price_2 * 0.98:
                trend_2 = "down"
            else:
                trend_2 = "stable"

        # Assemble market data
        market_data = {
            'condition_id': condition_id,
            'question': row['question'],
            'answer1': row['answer1'],
            'answer2': row['answer2'],

            # Token 1 (YES) data
            'token1': {
                'token_id': str(row['token1']),
                'answer': row['answer1'],
                'mid_price': round(mid_price_1, 4) if mid_price_1 else None,
                'spread': round(spread_1, 4) if spread_1 else None,
                'best_bid': deets_token1['best_bid'],
                'best_ask': deets_token1['best_ask'],
                'orderbook_imbalance': round(imbalance_1, 2),
                'trend': trend_1,
                'position': {
                    'size': pos_1['size'],
                    'avg_price': pos_1['avgPrice'],
                    'pnl_percent': round(pnl_1, 2) if pnl_1 else None
                }
            },

            # Token 2 (NO) data
            'token2': {
                'token_id': str(row['token2']),
                'answer': row['answer2'],
                'mid_price': round(mid_price_2, 4) if mid_price_2 else None,
                'spread': round(spread_2, 4) if spread_2 else None,
                'best_bid': deets_token2['best_bid'],
                'best_ask': deets_token2['best_ask'],
                'orderbook_imbalance': round(imbalance_2, 2),
                'trend': trend_2,
                'position': {
                    'size': pos_2['size'],
                    'avg_price': pos_2['avgPrice'],
                    'pnl_percent': round(pnl_2, 2) if pnl_2 else None
                }
            },

            # Market metadata
            'metadata': {
                'volatility_3h': round(row.get('3_hour', 0), 4),
                'neg_risk': row.get('neg_risk', False),
                'trade_size': row.get('trade_size', 0),
                'max_size': row.get('max_size', row.get('trade_size', 0))
            }
        }

        return market_data

    def prepare_llm_input(self, markets_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare comprehensive input for LLM analysis.

        Args:
            markets_data: List of market analysis dicts from analyze_market()

        Returns:
            Dict ready to be formatted into LLM prompt
        """
        # Get portfolio summary
        portfolio_summary = self._get_portfolio_summary()

        # Format markets for LLM
        llm_input = {
            'timestamp': pd.Timestamp.utcnow().isoformat(),
            'portfolio': portfolio_summary,
            'markets': markets_data
        }

        return llm_input

    def _get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get summary of current portfolio state.

        Returns:
            Dict with portfolio metrics
        """
        # Count positions
        num_positions = 0
        total_exposure = 0
        unrealized_pnl = 0

        if hasattr(global_state, 'positions'):
            for token, pos_data in global_state.positions.items():
                if pos_data['size'] > 0:
                    num_positions += 1
                    total_exposure += pos_data['size'] * pos_data['avgPrice']

                    # Try to calculate PnL if we have current market price
                    # (simplified - would need current price lookup)

        # Get available balance (simplified - would need actual balance check)
        available_capital = 1000  # Placeholder

        return {
            'num_positions': num_positions,
            'total_exposure': round(total_exposure, 2),
            'available_capital': available_capital,
            'capital_utilization': round(total_exposure / (total_exposure + available_capital), 2) if total_exposure + available_capital > 0 else 0
        }
