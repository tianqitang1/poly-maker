"""
LLM Maker Bot - Main Entry Point

LLM-aided market making bot that uses AI for strategic decisions
and fast execution for order placement.

Architecture:
- Slow loop (30-60s): LLM analyzes markets and generates signals
- Fast loop (event-driven): Execute trades based on LLM signals
- Fallback: Pre-configured logic if LLM unavailable
"""

import gc
import time
import asyncio
import traceback
import threading
import os
import sys
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from poly_data.polymarket_client import PolymarketClient
from poly_data.data_utils import update_markets, update_positions, update_orders
from poly_data.websocket_handlers import connect_market_websocket, connect_user_websocket
import poly_data.global_state as global_state
from poly_data.data_processing import remove_from_performing
from poly_utils.logging_utils import get_logger
from poly_utils.proxy_config import setup_proxy
from dotenv import load_dotenv

# Import LLM maker modules
from market_analyzer import MarketAnalyzer
from llm_strategy import LLMStrategy
from execution import ExecutionEngine

load_dotenv()

# Setup proxy configuration BEFORE initializing any API clients
setup_proxy(verbose=True)

# Global instances
market_analyzer = None
llm_strategy = None
execution_engine = None
config = None


def load_config():
    """Load configuration from YAML file."""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def update_once():
    """Initialize the application state by fetching market data, positions, and orders."""
    update_markets()    # Get market information from Google Sheets
    update_positions()  # Get current positions from Polymarket
    update_orders()     # Get current orders from Polymarket


def remove_from_pending():
    """Clean up stale trades that have been pending for too long (>10 seconds)."""
    try:
        current_time = time.time()
        removed_any = False

        # Iterate through all performing trades
        for col in list(global_state.performing.keys()):
            for trade_id in list(global_state.performing[col]):
                try:
                    # If trade has been pending for more than 10 seconds, remove it
                    timestamp = global_state.performing_timestamps.get(col, {}).get(trade_id, current_time)
                    time_pending = current_time - timestamp

                    if time_pending > 10:
                        print(f"⚠️  Removing stale trade {trade_id} from {col} (pending for {time_pending:.1f}s)")
                        remove_from_performing(col, trade_id)
                        removed_any = True
                except Exception as ex:
                    print(f"Error removing stale trade {trade_id} from {col}: {ex}")

        if removed_any:
            print("Cleaned up stale trades")

            # Force a position refresh
            try:
                update_positions(avgOnly=False)
            except Exception as ex:
                print(f"Error refreshing positions after cleanup: {ex}")

    except Exception as ex:
        print(f"Error in remove_from_pending: {ex}")
        traceback.print_exc()


def update_periodically():
    """
    Background thread function that periodically updates market data, positions and orders.
    - Positions and orders are updated every 5 seconds
    - Market data is updated every 30 seconds (every 6 cycles)
    - Stale pending trades are removed each cycle
    """
    i = 1
    while True:
        time.sleep(5)  # Update every 5 seconds

        try:
            # Clean up stale trades
            remove_from_pending()

            # Update positions and orders every cycle
            update_positions(avgOnly=True)  # Only update average price, not position size
            update_orders()

            # Update market data every 6th cycle (30 seconds)
            if i % 6 == 0:
                update_markets()
                i = 1

            gc.collect()  # Force garbage collection to free memory
            i += 1
        except:
            print("Error in update_periodically")
            traceback.print_exc()


def llm_decision_loop():
    """
    Background thread that runs LLM analysis periodically.
    This is the "slow loop" that generates trading signals.
    """
    global market_analyzer, llm_strategy, execution_engine, config

    logger = get_logger('llm_maker.decision_loop')
    logger.info("LLM decision loop started")

    decision_interval = config['strategy']['decision_interval']
    max_markets = config['strategy']['max_markets_per_query']

    while True:
        try:
            # Wait for next decision interval
            time.sleep(decision_interval)

            logger.info("Running LLM analysis...")

            # Get top markets to analyze
            top_markets = market_analyzer.get_top_markets(max_markets=max_markets)

            if not top_markets:
                logger.warning("No markets to analyze")
                continue

            logger.info(f"Analyzing {len(top_markets)} markets")

            # Analyze each market
            markets_data = []
            for market_id in top_markets:
                market_data = market_analyzer.analyze_market(market_id)
                if market_data:
                    markets_data.append(market_data)

            if not markets_data:
                logger.warning("No valid market data collected")
                continue

            # Prepare LLM input
            llm_input = market_analyzer.prepare_llm_input(markets_data)

            # Generate trading signals
            result = llm_strategy.generate_trading_signals(llm_input)

            if result['success']:
                signals = result['signals']
                logger.info(f"Generated {len(signals)} signals using {result['model_used']}")

                # Log some signal details
                for signal in signals[:3]:  # Log first 3
                    logger.info(
                        f"  {signal['action']} (confidence: {signal['confidence']:.2f}, "
                        f"priority: {signal['priority']}) - {signal['reasoning']}"
                    )

                # Update execution engine with new signals
                execution_engine.update_signals(signals)
            else:
                logger.error("Failed to generate trading signals")

            # Print stats
            stats = llm_strategy.get_stats()
            logger.info(f"LLM Stats: {stats['llm_client']}" if 'llm_client' in stats else "LLM Stats: N/A")

        except Exception as e:
            logger.error(f"Error in LLM decision loop: {e}")
            logger.debug(traceback.format_exc())


# Custom perform_trade function that uses LLM signals
async def perform_trade_with_llm(market):
    """
    Execute trades for a market based on LLM signals.
    This replaces the standard perform_trade function.
    """
    if execution_engine:
        await execution_engine.execute_market(market)
    await asyncio.sleep(1)


# Monkey-patch the trading module to use our LLM execution
def setup_llm_trading():
    """Replace standard trading logic with LLM-based execution."""
    import trading
    trading.perform_trade = perform_trade_with_llm
    print("LLM trading logic activated")


async def main():
    """Main application entry point."""
    global market_analyzer, llm_strategy, execution_engine, config

    # Initialize logger
    logger = get_logger('llm_maker')
    global_state.logger = logger

    # Load configuration
    logger.info("Loading configuration...")
    config = load_config()
    logger.info(f"LLM Provider: {config['llm']['provider']}")
    logger.info(f"Decision Interval: {config['strategy']['decision_interval']}s")

    # Initialize client with LLM_MAKER credentials
    pk = os.getenv('LLM_MAKER_PK')
    browser_address = os.getenv('LLM_MAKER_BROWSER_ADDRESS')

    if not pk or not browser_address:
        logger.error("Missing LLM_MAKER_PK or LLM_MAKER_BROWSER_ADDRESS environment variables!")
        print("\nERROR: Missing LLM_MAKER_PK or LLM_MAKER_BROWSER_ADDRESS environment variables!")
        print("Please add these to your .env file:")
        print("LLM_MAKER_PK=your_private_key_here")
        print("LLM_MAKER_BROWSER_ADDRESS=your_wallet_address_here")
        print("\nAlso ensure you have an LLM API key configured:")
        print("GEMINI_API_KEY=your_api_key_here  (or ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)")
        return

    global_state.client = PolymarketClient(private_key=pk, browser_address=browser_address)

    # Initialize state and fetch initial data
    global_state.all_tokens = []
    update_once()
    logger.info(f"After initial updates - Orders: {len(global_state.orders)}, Positions: {len(global_state.positions)}")

    logger.info("")
    logger.info(f'[LLM MAKER] There are {len(global_state.df)} markets, {len(global_state.positions)} positions and {len(global_state.orders)} orders.')

    # Initialize LLM maker components
    logger.info("Initializing LLM maker components...")
    market_analyzer = MarketAnalyzer(config)
    llm_strategy = LLMStrategy(config)
    execution_engine = ExecutionEngine(config)
    logger.info("LLM maker components initialized")

    # Setup LLM trading
    setup_llm_trading()

    # Start background update thread
    update_thread = threading.Thread(target=update_periodically, daemon=True)
    update_thread.start()
    logger.info("Started market data update thread")

    # Start LLM decision loop thread
    llm_thread = threading.Thread(target=llm_decision_loop, daemon=True)
    llm_thread.start()
    logger.info("Started LLM decision thread")

    # Give LLM thread time to generate first signals
    logger.info("Waiting for initial LLM analysis...")
    await asyncio.sleep(5)

    # Main loop - maintain websocket connections
    while True:
        try:
            # Connect to market and user websockets simultaneously
            await asyncio.gather(
                connect_market_websocket(global_state.all_tokens),
                connect_user_websocket()
            )
            print("Reconnecting to the websocket")
        except:
            print("Error in main loop")
            traceback.print_exc()

        await asyncio.sleep(1)
        gc.collect()  # Clean up memory


if __name__ == "__main__":
    asyncio.run(main())
