#!/usr/bin/env python3
"""
Near-Sure Trading Utility - Main Entry Point

A specialized trading utility for Polymarket that focuses on near-certain markets.

Usage:
    python -m near_sure.main trade          # Interactive trading mode
    python -m near_sure.main monitor        # Continuous risk monitoring
    python -m near_sure.main trade-monitor  # Trade then monitor
    python -m near_sure.main --help         # Show help

Features:
- Scans all markets for near-certain outcomes (price close to 0 or 1)
- Interactive market selection with arrow key navigation
- Passive order placement at existing bid prices near midpoint
- Account-wide stop-loss monitoring
"""

import sys
import argparse
from typing import Optional
from dotenv import load_dotenv

# Import components
from poly_data.polymarket_client import PolymarketClient
from near_sure.market_scanner import NearSureMarketScanner
from near_sure.interactive_ui import InteractiveMarketSelector
from near_sure.order_manager import NearSureOrderManager
from near_sure.risk_manager import NearSureRiskManager
from poly_utils.logging_utils import get_logger
from poly_utils.proxy_config import setup_proxy

# Load environment variables and setup proxy
load_dotenv()
setup_proxy(verbose=False)


def trade_mode(dry_run: bool = False):
    """
    Interactive trading mode - scan, select, and place orders.

    Args:
        dry_run: If True, simulate orders without placing them
    """
    print(f"\n{'='*80}")
    print("NEAR-SURE TRADING UTILITY")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE TRADING'}")
    print(f"{'='*80}\n")

    try:
        # Initialize logger
        logger = get_logger('near_sure')

        # Initialize client with near-sure credentials
        client = PolymarketClient(use_near_sure=True)

        # Initialize components
        scanner = NearSureMarketScanner(client)
        ui = InteractiveMarketSelector(scanner)
        order_manager = NearSureOrderManager(client, logger=logger)

        # Run interactive session
        configured_trades = ui.run_interactive_session()

        if not configured_trades:
            print("\nNo trades configured. Exiting.")
            return

        # Place orders
        print(f"\n{'='*80}")
        print(f"Placing {len(configured_trades)} order(s)...")
        print(f"{'='*80}\n")

        results = order_manager.place_batch_orders(
            configured_trades=configured_trades,
            dry_run=dry_run,
            delay_between_orders=1.0
        )

        # Display final summary
        if results['successful']:
            print("\n✓ Successfully placed orders:")
            for trade in results['successful']:
                print(f"  - {trade['market'][:60]} (${trade['amount']:.2f})")

        if results['failed']:
            print("\n✗ Failed orders:")
            for trade in results['failed']:
                print(f"  - {trade['market'][:60]} (${trade['amount']:.2f})")

        print(f"\nTotal: {results['total']} | Success: {len(results['successful'])} | Failed: {len(results['failed'])}")

    except ValueError as e:
        # Only show the env var hint for the explicit missing-env error that
        # PolymarketClient raises. For any other ValueError, surface the real
        # traceback so we can debug issues (e.g., interactive prompt errors)
        # instead of mis-reporting them as missing env vars.
        msg = str(e)
        if "Missing environment variables" in msg:
            print(f"\n❌ Configuration Error: {msg}")
            print("\nPlease ensure you have set the following in your .env file:")
            print("  NEAR_SURE_PK=<your_private_key>")
            print("  NEAR_SURE_BROWSER_ADDRESS=<your_wallet_address>")
        else:
            import traceback
            print("\n❌ Error (debug details below):")
            traceback.print_exc()
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def monitor_mode(
    check_interval: int = 30,
    stop_loss_pct: float = -0.10,
    dry_run: bool = False
):
    """
    Continuous risk monitoring mode.

    Args:
        check_interval: Seconds between position checks
        stop_loss_pct: Stop loss percentage (e.g., -0.10 for -10%)
        dry_run: If True, simulate stop losses without executing
    """
    print(f"\n{'='*80}")
    print("NEAR-SURE RISK MONITORING")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE MONITORING'}")
    print(f"{'='*80}\n")

    try:
        # Initialize logger
        logger = get_logger('near_sure')

        # Initialize client with near-sure credentials
        client = PolymarketClient(use_near_sure=True)

        # Initialize risk manager
        risk_manager = NearSureRiskManager(
            client=client,
            stop_loss_pct=stop_loss_pct,
            logger=logger
        )

        # Start continuous monitoring
        risk_manager.continuous_monitoring(
            check_interval=check_interval,
            dry_run=dry_run
        )

    except ValueError as e:
        msg = str(e)
        if "Missing environment variables" in msg:
            print(f"\n❌ Configuration Error: {msg}")
            print("\nPlease ensure you have set the following in your .env file:")
            print("  NEAR_SURE_PK=<your_private_key>")
            print("  NEAR_SURE_BROWSER_ADDRESS=<your_wallet_address>")
        else:
            import traceback
            print("\n❌ Error (debug details below):")
            traceback.print_exc()
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def trade_and_monitor_mode(
    check_interval: int = 30,
    stop_loss_pct: float = -0.10,
    dry_run: bool = False
):
    """
    Combined mode - place trades then start monitoring.

    Args:
        check_interval: Seconds between position checks
        stop_loss_pct: Stop loss percentage
        dry_run: If True, simulate without executing
    """
    # First, place trades
    trade_mode(dry_run=dry_run)

    # Ask if user wants to start monitoring
    try:
        from InquirerPy import inquirer
        start_monitoring = inquirer.confirm(
            message="Start continuous risk monitoring?",
            default=True
        ).execute()

        if start_monitoring:
            print("\n" + "="*80)
            print("Transitioning to monitoring mode...")
            print("="*80 + "\n")
            monitor_mode(
                check_interval=check_interval,
                stop_loss_pct=stop_loss_pct,
                dry_run=dry_run
            )
        else:
            print("\nSkipping monitoring. Exiting.")

    except ImportError:
        # InquirerPy not available, default to monitoring
        print("\nStarting monitoring mode...")
        monitor_mode(
            check_interval=check_interval,
            stop_loss_pct=stop_loss_pct,
            dry_run=dry_run
        )


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Near-Sure Trading Utility for Polymarket",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive trading mode
  python -m near_sure.main trade

  # Dry run (simulate without placing orders)
  python -m near_sure.main trade --dry-run

  # Monitor positions with custom stop loss
  python -m near_sure.main monitor --stop-loss -0.15

  # Trade and then monitor
  python -m near_sure.main trade-monitor

Environment Variables:
  NEAR_SURE_PK              Private key for near-sure account
  NEAR_SURE_BROWSER_ADDRESS Wallet address for near-sure account
        """
    )

    parser.add_argument(
        'mode',
        choices=['trade', 'monitor', 'trade-monitor'],
        help='Operation mode'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate without placing orders or executing stop losses'
    )

    parser.add_argument(
        '--check-interval',
        type=int,
        default=30,
        help='Seconds between position checks in monitor mode (default: 30)'
    )

    parser.add_argument(
        '--stop-loss',
        type=float,
        default=-0.10,
        help='Stop loss percentage, e.g., -0.10 for -10%% (default: -0.10)'
    )

    args = parser.parse_args()

    # Validate stop loss
    if args.stop_loss >= 0:
        print("Error: Stop loss must be negative (e.g., -0.10 for -10%%)")
        sys.exit(1)

    # Route to appropriate mode
    if args.mode == 'trade':
        trade_mode(dry_run=args.dry_run)

    elif args.mode == 'monitor':
        monitor_mode(
            check_interval=args.check_interval,
            stop_loss_pct=args.stop_loss,
            dry_run=args.dry_run
        )

    elif args.mode == 'trade-monitor':
        trade_and_monitor_mode(
            check_interval=args.check_interval,
            stop_loss_pct=args.stop_loss,
            dry_run=args.dry_run
        )


if __name__ == '__main__':
    main()
