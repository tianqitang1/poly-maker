"""
OG Maker Kai Bot - Enhanced Original Strategy
Uses the original market making strategy with improvements:
- No reverse position check (allows two-sided making)
- Volatility-adjusted take-profit
"""
import gc                      # Garbage collection
import time                    # Time functions
import asyncio                 # Asynchronous I/O
import traceback               # Exception handling
import threading               # Thread management
import os
import sys

# Add parent directory to path to import shared modules
# Insert at index 1 to keep current directory at index 0
# This ensures 'import trading' picks up the local trading.py, not the one in parent
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from poly_data.polymarket_client import PolymarketClient
from poly_data.data_utils import update_markets, update_positions, update_orders
from poly_data.websocket_handlers import connect_market_websocket, connect_user_websocket
import poly_data.global_state as global_state
from poly_data.data_processing import remove_from_performing
from poly_utils.logging_utils import get_logger
from poly_utils.proxy_config import setup_proxy
from dotenv import load_dotenv

load_dotenv()

# Setup proxy configuration BEFORE initializing any API clients
setup_proxy(verbose=True)

def update_once():
    """
    Initialize the application state by fetching market data, positions, and orders.
    """
    update_markets()    # Get market information from Google Sheets
    update_positions()  # Get current positions from Polymarket
    update_orders()     # Get current orders from Polymarket

def remove_from_pending():
    """
    Clean up stale trades that have been pending for too long (>15 seconds).
    This prevents the system from getting stuck on trades that may have failed.
    """
    try:
        current_time = time.time()

        # Iterate through all performing trades
        for col in list(global_state.performing.keys()):
            for trade_id in list(global_state.performing[col]):

                try:
                    # If trade has been pending for more than 15 seconds, remove it
                    if current_time - global_state.performing_timestamps[col].get(trade_id, current_time) > 15:
                        print(f"Removing stale entry {trade_id} from {col} after 15 seconds")
                        remove_from_performing(col, trade_id)
                        print("After removing: ", global_state.performing, global_state.performing_timestamps)
                except:
                    print("Error in remove_from_pending")
                    print(traceback.format_exc())
    except:
        print("Error in remove_from_pending")
        print(traceback.format_exc())

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
            print(traceback.format_exc())

async def main():
    """
    Main application entry point. Initializes client, data, and manages websocket connections.
    """
    # Initialize logger
    logger = get_logger('og_maker_kai')
    global_state.logger = logger

    # Initialize client with OG_MAKER credentials
    pk = os.getenv('OG_MAKER_PK')
    browser_address = os.getenv('OG_MAKER_BROWSER_ADDRESS')

    if not pk or not browser_address:
        logger.error("Missing OG_MAKER_PK or OG_MAKER_BROWSER_ADDRESS environment variables!")
        print("\nERROR: Missing OG_MAKER_PK or OG_MAKER_BROWSER_ADDRESS environment variables!")
        print("Please add these to your .env file:")
        print("OG_MAKER_PK=your_private_key_here")
        print("OG_MAKER_BROWSER_ADDRESS=your_wallet_address_here")
        return

    global_state.client = PolymarketClient(private_key=pk, browser_address=browser_address)

    # Initialize state and fetch initial data
    global_state.all_tokens = []
    update_once()
    logger.info(f"After initial updates - Orders: {len(global_state.orders)}, Positions: {len(global_state.positions)}")

    logger.info("")
    logger.info(f'[OG MAKER] There are {len(global_state.df)} markets, {len(global_state.positions)} positions and {len(global_state.orders)} orders.')
    logger.debug(f'Starting positions: {global_state.positions}')

    # Start background update thread
    update_thread = threading.Thread(target=update_periodically, daemon=True)
    update_thread.start()

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
            print(traceback.format_exc())

        await asyncio.sleep(1)
        gc.collect()  # Clean up memory

if __name__ == "__main__":
    asyncio.run(main())
