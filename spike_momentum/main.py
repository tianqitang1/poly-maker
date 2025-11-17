#!/usr/bin/env python3
"""
Spike Momentum Trading Bot - Main Entry Point

A sophisticated trading bot that identifies price spikes driven by real events
by combining technical analysis, news monitoring, and LLM-powered context analysis.

Usage:
    python -m spike_momentum.main scan          # Observe spikes (no trading)
    python -m spike_momentum.main trade         # Manual trading mode
    python -m spike_momentum.main auto          # Automated trading
    python -m spike_momentum.main --help        # Show help

Features:
- Real-time spike detection via WebSocket
- Sports news integration
- LLM analysis (Gemini/Claude/GPT) to validate spikes
- Focus on near-resolution markets
"""

import sys
import argparse
import time
import yaml
from datetime import datetime
from dotenv import load_dotenv

from poly_utils.logging_utils import get_logger
from poly_utils.proxy_config import setup_proxy
from spike_momentum.news_monitor import SportsNewsMonitor
from spike_momentum.spike_detector import SpikeDetector
from spike_momentum.llm_provider import LLMProvider
from spike_momentum.llm_analyzer import LLMAnalyzer

# Load environment variables and setup proxy
load_dotenv()
setup_proxy(verbose=False)


def scan_mode(config_path='spike_momentum/config.yaml', dry_run=True):
    """
    Scan mode - observe spikes without trading.

    Args:
        config_path: Path to configuration file
        dry_run: Always True for scan mode
    """
    print(f"\n{'='*120}")
    print("SPIKE MOMENTUM BOT - SCAN MODE")
    print("Observing markets for spikes (no trading)")
    print(f"{'='*120}\n")

    try:
        logger = get_logger('spike_momentum')
        logger.info("Starting scan mode...")

        # Load configuration
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"❌ Config file not found: {config_path}")
            print(f"   Please copy config.yaml.example to config.yaml and customize it.")
            sys.exit(1)

        # Initialize components
        print("Initializing components...")

        # News monitor
        news_monitor = SportsNewsMonitor(config.get('news', {}))
        print(f"✓ News monitor initialized")

        # Spike detector
        spike_detector = SpikeDetector(config.get('spike_detection', {}))
        print(f"✓ Spike detector initialized")

        # LLM analyzer (optional - only if enabled and API key present)
        llm_analyzer = None
        if config.get('llm', {}).get('enabled', True):
            try:
                llm_provider = LLMProvider(config['llm'])
                llm_analyzer = LLMAnalyzer(llm_provider, config['llm'])
                print(f"✓ LLM analyzer initialized ({config['llm']['provider']})")
            except Exception as e:
                print(f"⚠ LLM analyzer disabled: {e}")
                print(f"  Will run without LLM analysis (technical signals only)")

        print(f"\n{'='*120}")
        print("MONITORING STARTED")
        print(f"{'='*120}\n")
        print("Fetching initial news...")

        # Fetch initial news
        news_items = news_monitor.fetch_news(max_items=20)
        print(f"✓ Loaded {len(news_items)} recent news items\n")

        if news_items:
            print("Recent headlines:")
            for item in news_items[:5]:
                age_min = (datetime.now() - item.published).total_seconds() / 60
                print(f"  [{item.source}] {item.title[:80]}... ({age_min:.0f}m ago)")
            print()

        print("Waiting for price spikes...")
        print("(This is a demo mode - in production this would connect to live WebSocket data)\n")

        # Demo mode: simulate some price updates
        # In production, this would integrate with your existing WebSocket infrastructure
        demo_markets = [
            {'id': 'lakers_vs_warriors', 'question': 'Will the Lakers beat the Warriors tonight?'},
            {'id': 'chiefs_vs_bills', 'question': 'Will the Chiefs beat the Bills?'},
        ]

        scan_count = 0
        news_refresh_interval = config.get('news', {}).get('refresh_interval', 60)
        last_news_fetch = time.time()

        while True:
            scan_count += 1
            current_time = datetime.now().strftime('%H:%M:%S')

            # Demo: Simulate price updates (replace with real WebSocket data)
            print(f"[{current_time}] Scan #{scan_count} - Monitoring {len(demo_markets)} markets...")

            # Refresh news periodically
            if time.time() - last_news_fetch > news_refresh_interval:
                print(f"  Refreshing news...")
                news_items = news_monitor.fetch_news(max_items=20)
                print(f"  ✓ {len(news_items)} news items")
                last_news_fetch = time.time()

            # In production: Process real price updates from WebSocket
            # For now, just simulate
            for market in demo_markets:
                # Simulate price (in production, get from WebSocket)
                import random
                bid = 0.50 + random.uniform(-0.05, 0.05)
                ask = bid + 0.01

                # Update spike detector
                spike = spike_detector.update_price(
                    market_id=market['id'],
                    market_question=market['question'],
                    best_bid=bid,
                    best_ask=ask
                )

                if spike:
                    print(f"\n{'='*120}")
                    print(f"🚨 SPIKE DETECTED!")
                    print(f"{'='*120}")
                    print(f"Market: {spike.market_question}")
                    print(f"Direction: {spike.direction.upper()}")
                    print(f"Price change: {spike.price_change_pct:+.2f}% ({spike.previous_price:.3f} → {spike.current_price:.3f})")
                    print(f"Time window: {spike.time_window}s")
                    print(f"Spike strength: {spike.spike_strength:.2f} std devs")

                    # Find relevant news
                    news_matches = news_monitor.match_to_market(spike.market_question, max_results=3)

                    if news_matches:
                        print(f"\nRelated news ({len(news_matches)} items):")
                        for match in news_matches:
                            item = match['news']
                            print(f"  [{item.source}] {item.title}")
                            print(f"    Relevance: {match['relevance_score']:.2f}")

                    # LLM analysis (if available)
                    if llm_analyzer and news_matches:
                        print(f"\nRunning LLM analysis...")
                        analysis = llm_analyzer.analyze_spike(
                            market_question=spike.market_question,
                            current_price=spike.current_price,
                            previous_price=spike.previous_price,
                            price_change_pct=spike.price_change_pct,
                            news_items=[m['news'] for m in news_matches]
                        )

                        if analysis:
                            print(f"\n📊 LLM Analysis:")
                            print(f"  Justified: {'YES' if analysis.justified else 'NO'}")
                            print(f"  Confidence: {analysis.confidence}%")
                            print(f"  Recommendation: {analysis.recommendation.upper()}")
                            print(f"  Reasoning: {analysis.reasoning}")

                            if analysis.near_resolution:
                                print(f"  ⏰ Near resolution: {analysis.estimated_time_to_resolution}")

                            should_trade = llm_analyzer.should_trade(analysis)
                            print(f"\n  {'✅ WOULD TRADE' if should_trade else '❌ SKIP'} (in live mode)")

                    print(f"{'='*120}\n")

            # Wait before next scan
            time.sleep(10)

    except KeyboardInterrupt:
        print("\n\nScan stopped by user.")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def trade_mode(config_path='spike_momentum/config.yaml', dry_run=False):
    """
    Trade mode - detect spikes and execute trades (with confirmation).

    Args:
        config_path: Path to configuration file
        dry_run: If True, simulate without executing
    """
    print(f"\n{'='*120}")
    print(f"SPIKE MOMENTUM BOT - TRADE MODE")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE TRADING'}")
    print(f"{'='*120}\n")

    try:
        logger = get_logger('spike_momentum')
        logger.info(f"Starting trade mode (dry_run={dry_run})...")

        print("Trade mode is under development.")
        print("This will detect spikes, analyze with LLM, and prompt for trade confirmation.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def auto_mode(config_path='spike_momentum/config.yaml', dry_run=False):
    """
    Automated mode - fully automated spike detection and trading.

    Args:
        config_path: Path to configuration file
        dry_run: If True, simulate without executing
    """
    print(f"\n{'='*120}")
    print("SPIKE MOMENTUM BOT - AUTO MODE")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE TRADING'}")
    print(f"{'='*120}\n")
    print("Press Ctrl+C to stop\n")

    try:
        logger = get_logger('spike_momentum')
        logger.info(f"Starting auto mode (dry_run={dry_run})...")

        print("Auto mode is under development.")
        print("This will run fully automated spike detection, LLM analysis, and trading.")

        while True:
            time.sleep(10)

    except KeyboardInterrupt:
        print("\n\nAuto mode stopped by user.")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Spike Momentum Trading Bot for Polymarket",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan for spikes (observation only)
  python -m spike_momentum.main scan

  # Trade mode with confirmation (dry run)
  python -m spike_momentum.main trade --dry-run

  # Automated trading (live)
  python -m spike_momentum.main auto

  # Use custom config
  python -m spike_momentum.main scan --config my_config.yaml

Environment Variables:
  SPIKE_MOMENTUM_PK              Private key for spike bot account
  SPIKE_MOMENTUM_BROWSER_ADDRESS Wallet address for spike bot account
  GEMINI_API_KEY                 Google Gemini API key (recommended)
  ANTHROPIC_API_KEY              Anthropic Claude API key (alternative)
  OPENAI_API_KEY                 OpenAI API key (alternative)

Configuration:
  Edit spike_momentum/config.yaml to adjust:
  - LLM provider (Gemini Flash recommended)
  - Spike detection thresholds
  - News sources
  - Position sizing
  - Risk limits
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
        default='spike_momentum/config.yaml',
        help='Path to configuration file (default: spike_momentum/config.yaml)'
    )

    args = parser.parse_args()

    # Route to appropriate mode
    if args.mode == 'scan':
        scan_mode(config_path=args.config, dry_run=True)  # scan is always dry-run

    elif args.mode == 'trade':
        trade_mode(config_path=args.config, dry_run=args.dry_run)

    elif args.mode == 'auto':
        auto_mode(config_path=args.config, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
