"""
Interactive UI for Near-Sure Market Selection

Provides a terminal-based interactive interface for selecting markets and
configuring trade parameters using InquirerPy.
"""

import os
from typing import List, Dict, Tuple
import pandas as pd

try:
    from InquirerPy import inquirer
    from InquirerPy.base.control import Choice
    from InquirerPy.separator import Separator
    INQUIRER_AVAILABLE = True
except ImportError:
    INQUIRER_AVAILABLE = False
    print("Warning: InquirerPy not installed. Interactive mode will not be available.")
    print("Install with: pip install InquirerPy")


class InteractiveMarketSelector:
    """
    Interactive terminal UI for selecting markets and configuring trades.
    """

    def __init__(self, scanner):
        """
        Initialize the interactive selector.

        Args:
            scanner: NearSureMarketScanner instance
        """
        self.scanner = scanner

        if not INQUIRER_AVAILABLE:
            raise ImportError(
                "InquirerPy is required for interactive mode. "
                "Install with: pip install InquirerPy"
            )

    def select_markets(self, markets_df: pd.DataFrame) -> List[Dict]:
        """
        Interactive market selection interface.

        Args:
            markets_df: DataFrame of filtered markets

        Returns:
            List of selected market dictionaries
        """
        if markets_df.empty:
            print("No markets available for selection")
            return []

        # Create choices with formatted display
        choices = []

        choices.append(Separator("=== Near-Sure Markets (sorted by closing time) ==="))

        for idx, row in markets_df.iterrows():
            display_text = self.scanner.format_market_display(row.to_dict())
            choices.append(Choice(value=row.to_dict(), name=display_text))

        # Multi-select interface
        selected = inquirer.checkbox(
            message="Select markets to trade (Space to select, Enter to confirm):",
            choices=choices,
            instruction="↑↓: Navigate | Space: Select | Enter: Confirm",
            transformer=lambda result: f"{len(result)} market(s) selected",
            validate=lambda result: len(result) > 0,
            invalid_message="Please select at least one market",
            height=15,
            border=True,
        ).execute()

        return selected

    def get_trade_amount(self, market: Dict, default_amount: float = 100.0) -> float:
        """
        Prompt for trade amount for a specific market.

        Args:
            market: Market dictionary
            default_amount: Default trade size

        Returns:
            Trade amount in USDC
        """
        # Validate market structure
        if not isinstance(market, dict):
            raise ValueError(f"Invalid market data: expected dict, got {type(market)}")

        if 'question' not in market:
            raise ValueError(f"Invalid market data: missing 'question' key. Available keys: {list(market.keys())}")

        question = market.get('question', 'Unknown market')
        if not question or not isinstance(question, str):
            raise ValueError(f"Invalid market question: {question}")

        question_short = question[:70] + "..." if len(question) > 70 else question
        min_size = market.get('min_size', 10)
        # Default to the market minimum to minimize friction for the user.
        default_amt = float(min_size)
        simple_mode = os.getenv("INQUIRERPY_SIMPLE_PROMPT") or os.getenv("NEAR_SURE_SIMPLE_INPUT")

        def prompt_fallback() -> float:
            while True:
                raw = input(f"Trade amount (min {min_size} USDC) for '{question_short}': ").strip()
                try:
                    amount_val = float(raw)
                    if amount_val < min_size:
                        print(f"Amount must be at least {min_size} USDC. Try again.")
                        continue
                    return amount_val
                except ValueError:
                    print("Please enter a numeric amount (e.g., 100 or 75.5).")

        # If caller explicitly wants simple prompts, skip InquirerPy entirely.
        if simple_mode:
            return prompt_fallback()

        try:
            # Use a text prompt with validation instead of the number prompt,
            # which can raise ValueError on some terminals when parsing decimals.
            amount = inquirer.text(
                message=(
                    f"Trade amount (USDC) for:\n  {question_short}\n"
                    f"(min {min_size} USDC)"
                ),
                default=f"{default_amt:.2f}",
                validate=lambda x: self._validate_amount(x, min_size),
                transformer=lambda x: f"${x}",
            ).execute()
            return float(amount)
        except Exception as e:
            # Some terminals/tty setups make InquirerPy blow up. Fall back to a
            # simple stdin prompt so we don't crash mid-session.
            print(f"\n⚠ Interactive prompt failed: {e}")
            print("Falling back to basic input for this amount.\n")
            return prompt_fallback()

    @staticmethod
    def _validate_amount(value: str, min_size: float) -> bool:
        """Validate that value is a number >= min_size."""
        try:
            return float(value) >= min_size
        except Exception:
            return False

    def configure_trades(self, selected_markets: List[Dict]) -> List[Tuple[Dict, float]]:
        """
        Configure trade amounts for each selected market.

        Args:
            selected_markets: List of selected market dictionaries

        Returns:
            List of tuples (market, trade_amount)
        """
        trades = []

        print(f"\n{'='*80}")
        print(f"Configuring trade amounts for {len(selected_markets)} market(s)")
        print(f"{'='*80}\n")

        for idx, market in enumerate(selected_markets, 1):
            print(f"[{idx}/{len(selected_markets)}]")
            amount = self.get_trade_amount(market)
            trades.append((market, amount))
            print()

        return trades

    def confirm_trades(self, configured_trades: List[Tuple[Dict, float]]) -> bool:
        """
        Show summary and ask for final confirmation.

        Args:
            configured_trades: List of (market, amount) tuples

        Returns:
            True if confirmed, False otherwise
        """
        print(f"\n{'='*80}")
        print("TRADE SUMMARY")
        print(f"{'='*80}\n")

        total_capital = 0
        for idx, (market, amount) in enumerate(configured_trades, 1):
            question_short = market['question'][:60] + "..." if len(market['question']) > 60 else market['question']

            # Determine side
            if market['midpoint'] >= 0.85:
                side = "YES"
                price = market['midpoint']
            else:
                side = "NO"
                price = 1 - market['midpoint']

            print(f"{idx}. {question_short}")
            print(f"   Side: {side} @ {price:.3f} | Amount: ${amount:.2f} | Min Size: ${market.get('min_size', 10):.2f}")
            print(f"   Closes in: {market['hours_until_close']:.1f} hours\n")

            total_capital += amount

        print(f"{'='*80}")
        print(f"Total Capital Required: ${total_capital:.2f}")
        print(f"{'='*80}\n")

        confirmed = inquirer.confirm(
            message="Proceed with these trades?",
            default=False,
        ).execute()

        return confirmed

    def get_filter_parameters(self) -> Dict:
        """
        Interactive interface for setting market filter parameters.

        Returns:
            Dictionary of filter parameters
        """
        print(f"\n{'='*80}")
        print("Configure Market Filters")
        print(f"{'='*80}\n")

        def prompt_float(message: str, default: float, min_val: float = None, max_val: float = None) -> float:
            """Text-based float prompt with range validation to avoid number-widget quirks."""
            def _validate(x: str) -> bool:
                try:
                    val = float(x)
                except Exception:
                    return False
                if min_val is not None and val < min_val:
                    return False
                if max_val is not None and val > max_val:
                    return False
                return True

            while True:
                try:
                    val = inquirer.text(
                        message=message,
                        default=f"{float(default):.2f}",
                        validate=_validate,
                    ).execute()
                    return float(val)
                except Exception as e:
                    print(f"\n⚠ Prompt failed: {e}")
                    # Fallback to stdin if InquirerPy has issues
                    raw = input(f"{message} (default {default}): ").strip() or str(default)
                    try:
                        val = float(raw)
                        if (min_val is None or val >= min_val) and (max_val is None or val <= max_val):
                            return val
                    except Exception:
                        pass
                    print("Invalid input. Please try again.")

        min_midpoint = prompt_float(
            "Minimum midpoint price (for bullish near-sure):",
            default=0.85,
            min_val=0.5,
            max_val=0.99,
        )

        max_midpoint = prompt_float(
            "Maximum midpoint price (for bearish near-sure):",
            default=0.15,
            min_val=0.01,
            max_val=0.5,
        )

        max_hours = prompt_float(
            "Maximum hours until close (0 = no limit):",
            default=48,
            min_val=0,
        )

        min_hours = prompt_float(
            "Minimum hours until close:",
            default=1,
            min_val=0.1,
        )

        min_certainty = prompt_float(
            "Minimum certainty score (0-1):",
            default=0.70,
            min_val=0,
            max_val=1,
        )

        return {
            'min_midpoint': float(min_midpoint),
            'max_midpoint': float(max_midpoint),
            'max_hours_until_close': float(max_hours) if float(max_hours) > 0 else None,
            'min_hours_until_close': float(min_hours),
            'min_certainty': float(min_certainty),
        }

    def run_interactive_session(self) -> List[Tuple[Dict, float]]:
        """
        Run a complete interactive trading session.

        Returns:
            List of configured trades (market, amount) or empty list if cancelled
        """
        print(f"\n{'='*80}")
        print("NEAR-SURE TRADING - Interactive Mode")
        print(f"{'='*80}\n")

        # Get filter parameters
        filters = self.get_filter_parameters()

        # Scan for markets
        print("\nScanning for near-sure markets...\n")
        markets_df = self.scanner.filter_near_sure_markets(**filters)

        if markets_df.empty:
            print("\nNo markets found matching your criteria.")
            return []

        # Select markets
        selected_markets = self.select_markets(markets_df)

        if not selected_markets:
            print("\nNo markets selected. Exiting.")
            return []

        # Configure trade amounts
        configured_trades = self.configure_trades(selected_markets)

        # Final confirmation
        if self.confirm_trades(configured_trades):
            print("\n✓ Trades confirmed!")
            return configured_trades
        else:
            print("\n✗ Trades cancelled.")
            return []
