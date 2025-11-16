#!/usr/bin/env python3
"""
Negative Risk Arbitrage Bot - Main Entry Point

A specialized arbitrage bot for Polymarket that exploits negative risk markets
where YES + NO < $1.00.

Usage:
    python -m neg_risk_arb.main scan          # Scan for opportunities
    python -m neg_risk_arb.main trade         # Execute one opportunity
    python -m neg_risk_arb.main auto          # Automated arbitrage hunting
    python -m neg_risk_arb.main --help        # Show help

Features:
- Automated scanning for YES + NO < $1 opportunities
- Atomic execution with partial fill protection
- Configurable via config.yaml
- Real-time profit tracking
"""

import sys
import argparse
import time
from datetime import datetime

from poly_data.polymarket_client import PolymarketClient
from neg_risk_arb.arbitrage_scanner import ArbitrageScanner
from neg_risk_arb.executor import ArbitrageExecutor
from neg_risk_arb.risk_manager import ArbitrageRiskManager


def scan_mode(config_path='neg_risk_arb/config.yaml'):
    """
    Scan for arbitrage opportunities and display them.

    Args:
        config_path: Path to configuration file
    """
    print(f"\n{'='*120}")
    print("NEGATIVE RISK ARBITRAGE SCANNER")
    print(f"{'='*120}\n")

    try:
        # Initialize client
        client = PolymarketClient(account_type='neg_risk_arb')

        # Initialize scanner
        scanner = ArbitrageScanner(client, config_path)

        # Scan for opportunities
        opportunities = scanner.scan_for_opportunities(verbose=True)

        # Display results
        scanner.display_opportunities(opportunities)

        return opportunities

    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\nPlease ensure you have set the following in your .env file:")
        print("  NEG_RISK_ARB_PK=<your_private_key>")
        print("  NEG_RISK_ARB_BROWSER_ADDRESS=<your_wallet_address>")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def trade_mode(config_path='neg_risk_arb/config.yaml', dry_run=False):
    """
    Execute a single arbitrage opportunity.

    Args:
        config_path: Path to configuration file
        dry_run: If True, simulate without executing
    """
    print(f"\n{'='*120}")
    print(f"NEGATIVE RISK ARBITRAGE - TRADE MODE")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE TRADING'}")
    print(f"{'='*120}\n")

    try:
        # Initialize client
        client = PolymarketClient(account_type='neg_risk_arb')

        # Initialize components
        scanner = ArbitrageScanner(client, config_path)
        executor = ArbitrageExecutor(client, config_path)
        risk_manager = ArbitrageRiskManager(client, config_path)

        # Check risk limits
        can_trade, reason = risk_manager.check_risk_limits()
        if not can_trade:
            print(f"❌ Cannot trade: {reason}")
            return

        # Scan for opportunities
        print("Scanning for arbitrage opportunities...\n")
        opportunities = scanner.scan_for_opportunities(verbose=False)

        if opportunities.empty:
            print("No arbitrage opportunities found.")
            return

        # Display top opportunities
        scanner.display_opportunities(opportunities, max_display=5)

        # Select opportunity
        if scanner.config.get('require_confirmation', True) and not dry_run:
            try:
                from InquirerPy import inquirer

                # Create choices
                choices = [
                    scanner.format_opportunity(row.to_dict())
                    for _, row in opportunities.head(10).iterrows()
                ]

                selected_idx = inquirer.select(
                    message="Select opportunity to execute:",
                    choices=list(enumerate(choices)),
                    transformer=lambda x: x[1] if isinstance(x, tuple) else x
                ).execute()

                if isinstance(selected_idx, tuple):
                    selected_idx = selected_idx[0]

                opportunity = opportunities.iloc[selected_idx].to_dict()

            except ImportError:
                # InquirerPy not available, use first opportunity
                print("\nInquirerPy not available - using best opportunity")
                opportunity = opportunities.iloc[0].to_dict()
        else:
            # Auto-select best opportunity
            opportunity = opportunities.iloc[0].to_dict()

        # Execute arbitrage
        print(f"\n{'='*120}")
        print("EXECUTING ARBITRAGE")
        print(f"{'='*120}\n")

        result = executor.execute_arbitrage(opportunity, dry_run=dry_run)

        # Handle partial fills if needed
        if result.get('partial_fill') and not dry_run:
            print("\nHandling partial fill...")
            rescue_result = risk_manager.handle_partial_fill(
                opportunity=opportunity,
                yes_pos=result.get('yes_position', 0),
                no_pos=result.get('no_position', 0),
                expected_size=opportunity['tradeable_size'],
                dry_run=dry_run
            )

            if rescue_result.get('success'):
                # Retry merge after rescue
                yes_pos, no_pos = executor.verify_fills(opportunity, opportunity['tradeable_size'], 2)
                merge_size = min(yes_pos, no_pos)
                if merge_size > 0:
                    executor.merge_positions(opportunity, merge_size, dry_run)

        # Record execution
        if not dry_run:
            risk_manager.record_execution(result)

        # Display result
        print(f"\n{'='*120}")
        if result.get('success'):
            print("✓ ARBITRAGE SUCCESSFUL")
            if 'realized_profit' in result:
                print(f"Profit: ${result['realized_profit']:.2f} ({result['profit_bps']:.1f}bp)")
        else:
            print("✗ ARBITRAGE FAILED")
            print(f"Error: {result.get('error', 'Unknown error')}")
        print(f"{'='*120}\n")

        # Show stats
        stats = risk_manager.get_stats()
        if stats['total_trades'] > 0:
            print(f"\nSession Stats:")
            print(f"  Total Trades: {stats['total_trades']}")
            print(f"  Win Rate: {stats['win_rate']*100:.1f}%")
            print(f"  Total Profit: ${stats['total_profit']:.2f}")
            print(f"  Daily P&L: ${stats['daily_pnl']:.2f}\n")

    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\nPlease ensure you have set the following in your .env file:")
        print("  NEG_RISK_ARB_PK=<your_private_key>")
        print("  NEG_RISK_ARB_BROWSER_ADDRESS=<your_wallet_address>")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def auto_mode(config_path='neg_risk_arb/config.yaml', dry_run=False):
    """
    Automated arbitrage hunting mode.

    Args:
        config_path: Path to configuration file
        dry_run: If True, simulate without executing
    """
    print(f"\n{'='*120}")
    print("NEGATIVE RISK ARBITRAGE - AUTO MODE")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE TRADING'}")
    print(f"{'='*120}\n")
    print("Press Ctrl+C to stop\n")

    try:
        # Initialize client
        client = PolymarketClient(account_type='neg_risk_arb')

        # Initialize components
        scanner = ArbitrageScanner(client, config_path)
        executor = ArbitrageExecutor(client, config_path)
        risk_manager = ArbitrageRiskManager(client, config_path)

        scan_interval = scanner.config.get('scan_interval_sec', 30)
        max_concurrent = scanner.config.get('max_concurrent_positions', 3)

        while True:
            print(f"\n{'='*120}")
            print(f"Scan at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*120}")

            # Check risk limits
            can_trade, reason = risk_manager.check_risk_limits()
            if not can_trade:
                print(f"⚠️  Trading paused: {reason}")
                print(f"Waiting {scan_interval}s before next scan...")
                time.sleep(scan_interval)
                continue

            # Scan for opportunities
            opportunities = scanner.scan_for_opportunities(verbose=False)

            if opportunities.empty:
                print("No arbitrage opportunities found.")
            else:
                print(f"Found {len(opportunities)} opportunities")
                scanner.display_opportunities(opportunities, max_display=3)

                # Execute best opportunity automatically
                opportunity = opportunities.iloc[0].to_dict()

                print("\nExecuting best opportunity...")
                result = executor.execute_arbitrage(opportunity, dry_run=dry_run)

                # Handle partial fills
                if result.get('partial_fill') and not dry_run:
                    rescue_result = risk_manager.handle_partial_fill(
                        opportunity=opportunity,
                        yes_pos=result.get('yes_position', 0),
                        no_pos=result.get('no_position', 0),
                        expected_size=opportunity['tradeable_size'],
                        dry_run=dry_run
                    )

                    if rescue_result.get('success'):
                        yes_pos, no_pos = executor.verify_fills(opportunity, opportunity['tradeable_size'], 2)
                        merge_size = min(yes_pos, no_pos)
                        if merge_size > 0:
                            executor.merge_positions(opportunity, merge_size, dry_run)

                # Record execution
                if not dry_run:
                    risk_manager.record_execution(result)

                # Show result
                if result.get('success'):
                    print(f"\n✓ Arbitrage successful! Profit: ${result.get('realized_profit', 0):.2f}")
                else:
                    print(f"\n✗ Arbitrage failed: {result.get('error', 'Unknown')}")

            # Show stats
            stats = risk_manager.get_stats()
            if stats['total_trades'] > 0:
                print(f"\nSession Stats: {stats['total_trades']} trades | "
                      f"Win Rate: {stats['win_rate']*100:.1f}% | "
                      f"Total P&L: ${stats['daily_pnl']:.2f}")

            print(f"\nNext scan in {scan_interval}s...")
            time.sleep(scan_interval)

    except KeyboardInterrupt:
        print("\n\nStopping automated arbitrage...")
        stats = risk_manager.get_stats()
        if stats['total_trades'] > 0:
            print(f"\nFinal Stats:")
            print(f"  Total Trades: {stats['total_trades']}")
            print(f"  Win Rate: {stats['win_rate']*100:.1f}%")
            print(f"  Total Profit: ${stats['total_profit']:.2f}")
        print("\nArbitrage bot stopped.")
        sys.exit(0)

    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\nPlease ensure you have set the following in your .env file:")
        print("  NEG_RISK_ARB_PK=<your_private_key>")
        print("  NEG_RISK_ARB_BROWSER_ADDRESS=<your_wallet_address>")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Negative Risk Arbitrage Bot for Polymarket",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan for arbitrage opportunities
  python -m neg_risk_arb.main scan

  # Execute one opportunity
  python -m neg_risk_arb.main trade

  # Dry run (simulate without trading)
  python -m neg_risk_arb.main trade --dry-run

  # Automated arbitrage hunting
  python -m neg_risk_arb.main auto

  # Use custom config file
  python -m neg_risk_arb.main scan --config my_config.yaml

Environment Variables:
  NEG_RISK_ARB_PK              Private key for arbitrage account
  NEG_RISK_ARB_BROWSER_ADDRESS Wallet address for arbitrage account

Configuration:
  Edit neg_risk_arb/config.yaml to adjust:
  - Minimum profit thresholds
  - Position sizing
  - Partial fill strategies
  - Risk limits
  - And more...
        """
    )

    parser.add_argument(
        'mode',
        choices=['scan', 'trade', 'auto'],
        help='Operation mode'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate without executing real trades'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='neg_risk_arb/config.yaml',
        help='Path to configuration file (default: neg_risk_arb/config.yaml)'
    )

    args = parser.parse_args()

    # Route to appropriate mode
    if args.mode == 'scan':
        scan_mode(config_path=args.config)

    elif args.mode == 'trade':
        trade_mode(config_path=args.config, dry_run=args.dry_run)

    elif args.mode == 'auto':
        auto_mode(config_path=args.config, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
