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
# Insert current directory at 0 to ensure local trading.py is preferred over root's
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Insert parent directory at 1 to allow importing poly_data
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from poly_data.polymarket_client import PolymarketClient
from poly_data.data_utils import update_positions, update_orders
from poly_data.websocket_handlers import connect_market_websocket, connect_user_websocket
import poly_data.global_state as global_state
from poly_data.data_processing import remove_from_performing
from poly_utils.logging_utils import get_logger
from poly_utils.proxy_config import setup_proxy
from poly_data.utils import get_sheet_df
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Initialize logger at module level
logger = get_logger('og_maker_kai')

# Setup proxy configuration BEFORE initializing any API clients
setup_proxy(verbose=True)

def update_markets_live():
    """
    Fetch and validate markets from Google Sheets.
    - Validates presence of required columns: question, max_size, trade_size, param_type
    - Updates global_state.df and global_state.params atomically
    - Rebuilds global_state.all_tokens preserving tokens with active positions/orders
    """
    try:
        previous_tokens = set(str(t) for t in global_state.all_tokens)
        received_df, received_params = get_sheet_df()
        
        if len(received_df) == 0:
            logger.warning("No markets found or empty sheet.")
            return

        # Validation: Required columns
        required_cols = ['question', 'max_size', 'trade_size', 'param_type']
        if not all(col in received_df.columns for col in required_cols):
            logger.error(f"Missing required columns: {[c for c in required_cols if c not in received_df.columns]}")
            return

        # Validation: Filter rows with missing values in required columns
        # Check for NaN and empty strings
        valid_mask = received_df[required_cols].notna().all(axis=1) & (received_df[required_cols] != "").all(axis=1)
        valid_df = received_df[valid_mask].copy()
        
        if len(valid_df) < len(received_df):
             logger.info(f"Filtered {len(received_df) - len(valid_df)} invalid rows. Keeping {len(valid_df)} valid markets.")

        if len(valid_df) == 0:
            logger.warning("No valid markets after filtering.")
            return

        # Ensure token columns are strings
        for col in ['token1', 'token2']:
            valid_df[col] = valid_df[col].astype(str)

        # Update global state atomically
        global_state.df = valid_df
        global_state.params = received_params

        # Rebuild all_tokens
        # We want to keep:
        # 1. Tokens in the new valid_df (active markets)
        # 2. Tokens where we have a position (size != 0)
        # 3. Tokens where we have an active order (size > 0)
        
        new_tokens = set()
        
        # 1. New tokens from valid markets
        for _, row in valid_df.iterrows():
             token1, token2 = str(row['token1']), str(row['token2'])
             new_tokens.add(token1)
             new_tokens.add(token2)
             
             # Update REVERSE_TOKENS
             global_state.REVERSE_TOKENS[token1] = token2
             global_state.REVERSE_TOKENS[token2] = token1
             
             # Initialize performing set if needed
             for col in [f"{token1}_buy", f"{token1}_sell", f"{token2}_buy", f"{token2}_sell"]:
                if col not in global_state.performing:
                    global_state.performing[col] = set()

        # 2. Existing positions (preserve monitoring for exits)
        for token, pos in global_state.positions.items():
            if pos.get('size', 0) != 0:
                new_tokens.add(str(token))

        # 3. Existing orders (preserve monitoring for execution)
        for token, orders in global_state.orders.items():
            if orders.get('buy', {}).get('size', 0) > 0 or orders.get('sell', {}).get('size', 0) > 0:
                new_tokens.add(str(token))

        # Update the global token list
        global_state.all_tokens = sorted(new_tokens)

        # Track token1/token2 per market for websocket orientation
        global_state.MARKET_TOKENS = {}
        for _, row in valid_df.iterrows():
            cid = str(row['condition_id'])
            global_state.MARKET_TOKENS[cid] = {'token1': str(row['token1']), 'token2': str(row['token2'])}

        tokens_changed = new_tokens != previous_tokens
        if tokens_changed:
            added = new_tokens - previous_tokens
            removed = previous_tokens - new_tokens
            global_state.all_tokens_version += 1
            logger.info(
                f"Updated active markets. Monitoring {len(global_state.all_tokens)} tokens "
                f"(+{len(added)} / -{len(removed)}; version {global_state.all_tokens_version})."
            )
        else:
            logger.info(f"Updated active markets. Monitoring {len(global_state.all_tokens)} tokens (no change).")

    except Exception as e:
        logger.error(f"Error in update_markets_live: {e}", exc_info=True)

def update_once():
    """
    Initialize the application state by fetching market data, positions, and orders.
    """
    update_markets_live()    # Get market information from Google Sheets
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
                        logger.info(f"Removing stale entry {trade_id} from {col} after 15 seconds")
                        remove_from_performing(col, trade_id)
                        logger.debug(f"After removing: {global_state.performing} {global_state.performing_timestamps}")
                except:
                    logger.error("Error in remove_from_pending loop", exc_info=True)
    except:
        logger.error("Error in remove_from_pending", exc_info=True)

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
                update_markets_live()
                i = 1

            gc.collect()  # Force garbage collection to free memory
            i += 1
        except:
            logger.error("Error in update_periodically", exc_info=True)

async def main():
    """
    Main application entry point. Initializes client, data, and manages websocket connections.
    """
    # Initialize global state logger (already initialized at module level, but setting explicitly for other modules)
    global_state.logger = logger

    # Initialize client with OG_MAKER account type
    # The PolymarketClient will handle loading the correct credentials from .env
    global_state.client = PolymarketClient(account_type='og_maker')

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
        market_task = asyncio.create_task(connect_market_websocket(global_state.all_tokens))
        user_task = asyncio.create_task(connect_user_websocket())

        done, pending = await asyncio.wait(
            [market_task, user_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.info("Cancelled remaining websocket task after restart request")

        for task in done:
            try:
                task.result()
            except Exception:
                logger.error("Websocket task ended with error", exc_info=True)

        logger.info("Reconnecting to the websocket")
        await asyncio.sleep(1)
        gc.collect()  # Clean up memory

if __name__ == "__main__":
    asyncio.run(main())
