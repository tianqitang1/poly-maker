import gc                      # Garbage collection
import time                    # Time functions
import asyncio                 # Asynchronous I/O
import traceback               # Exception handling
import threading               # Thread management

from poly_data.polymarket_client import PolymarketClient
from poly_data.data_utils import update_markets, update_positions, update_orders
from poly_data.websocket_handlers import connect_market_websocket, connect_user_websocket
import poly_data.global_state as global_state
from poly_data.data_processing import remove_from_performing
from poly_utils.logging_utils import get_logger
from dotenv import load_dotenv

load_dotenv()

def update_once():
    """
    Initialize the application state by fetching market data, positions, and orders.
    """
    update_markets()    # Get market information from Google Sheets
    update_positions()  # Get current positions from Polymarket
    update_orders()     # Get current orders from Polymarket

def cleanup_inactive_markets():
    """
    Clean up positions and orders for markets that are no longer in the trading list.
    This handles markets that lost rewards or were removed from the sheet.
    """
    try:
        # Get set of active condition_ids from loaded dataframe
        active_condition_ids = set(global_state.df['condition_id'].tolist())

        # Get all tokens we have positions or orders on
        active_tokens = set()
        for token in global_state.positions.keys():
            if global_state.positions[token]['size'] > 0:
                active_tokens.add(token)

        for token in global_state.orders.keys():
            if global_state.orders[token]['buy']['size'] > 0 or global_state.orders[token]['sell']['size'] > 0:
                active_tokens.add(token)

        # Find markets we have exposure to but aren't in the trading list
        # Build a mapping of all tokens to their condition_ids from the dataframe
        all_tokens_in_df = set()
        for _, row in global_state.df.iterrows():
            all_tokens_in_df.add(str(row['token1']))
            all_tokens_in_df.add(str(row['token2']))

        tokens_to_cleanup = []
        for token in active_tokens:
            token_str = str(token)
            # If this token is not in any of the active markets, mark for cleanup
            if token_str not in all_tokens_in_df:
                # Find the condition_id by looking up in REVERSE_TOKENS or positions
                condition_id = None
                # Try to find it in the reverse tokens mapping if available
                if hasattr(global_state, 'TOKEN_TO_CONDITION'):
                    condition_id = global_state.TOKEN_TO_CONDITION.get(token_str)

                tokens_to_cleanup.append((token, condition_id))

        if not tokens_to_cleanup:
            print("No inactive markets found with positions/orders")
            return

        print(f"\n{'='*80}")
        print(f"Found {len(tokens_to_cleanup)} tokens on inactive markets. Cleaning up...")
        print(f"{'='*80}")

        for token, condition_id in tokens_to_cleanup:
            token_str = str(token)
            position = global_state.positions.get(token_str, {'size': 0, 'avgPrice': 0})
            orders = global_state.orders.get(token_str, {'buy': {'size': 0, 'price': 0}, 'sell': {'size': 0, 'price': 0}})

            print(f"\nToken {token_str} (market {condition_id[:16]}...):")
            print(f"  Position: {position['size']} @ avg {position['avgPrice']}")
            print(f"  Orders: Buy {orders['buy']['size']} @ {orders['buy']['price']}, Sell {orders['sell']['size']} @ {orders['sell']['price']}")

            # Cancel all orders for this token
            if orders['buy']['size'] > 0 or orders['sell']['size'] > 0:
                print(f"  → Cancelling all orders for token {token_str}")
                global_state.client.cancel_all_asset(token)

            # Exit position if we have one
            if position['size'] > 0:
                print(f"  → Exiting position of {position['size']} shares")
                try:
                    # Try to get current market price to sell at best bid
                    sell_price = 0.5  # Default fallback price

                    if condition_id:
                        try:
                            from poly_data.trading_utils import get_best_bid_ask_deets
                            deets = get_best_bid_ask_deets(condition_id, 'token1', 20, 0.1)
                            best_bid = deets.get('best_bid')
                            if best_bid and best_bid > 0:
                                sell_price = best_bid
                        except:
                            pass  # Fall back to 0.5

                    print(f"  → Placing market sell order for {position['size']} @ {sell_price}")
                    global_state.client.create_order(
                        token,
                        'SELL',
                        sell_price,
                        position['size'],
                        False  # Assume not neg_risk since we don't have market info
                    )
                except Exception as ex:
                    print(f"  → Error exiting position: {ex}")
                    traceback.print_exc()

        print(f"\n{'='*80}")
        print("Cleanup complete")
        print(f"{'='*80}\n")

    except Exception as ex:
        print(f"Error in cleanup_inactive_markets: {ex}")
        print(traceback.format_exc())

def remove_from_pending():
    """
    Clean up stale trades that have been pending for too long (>10 seconds).
    This prevents the system from getting stuck on trades that may have failed or never received confirmation.
    Also triggers a position refresh after cleanup to ensure state is correct.
    """
    try:
        current_time = time.time()
        removed_any = False
        affected_tokens = set()

        # Iterate through all performing trades
        for col in list(global_state.performing.keys()):
            for trade_id in list(global_state.performing[col]):

                try:
                    # If trade has been pending for more than 10 seconds, remove it
                    timestamp = global_state.performing_timestamps.get(col, {}).get(trade_id, current_time)
                    time_pending = current_time - timestamp

                    if time_pending > 10:
                        print(f"⚠️  Removing stale trade {trade_id} from {col} (pending for {time_pending:.1f}s)")

                        # Extract token from col (format: "token_side")
                        token = col.rsplit('_', 1)[0]
                        affected_tokens.add(token)

                        remove_from_performing(col, trade_id)
                        removed_any = True
                except Exception as ex:
                    print(f"Error removing stale trade {trade_id} from {col}: {ex}")
                    traceback.print_exc()

        if removed_any:
            print(f"Cleaned up stale trades. Affected tokens: {affected_tokens}")
            print(f"Remaining performing: {global_state.performing}")

            # Force a position refresh for affected tokens
            try:
                from poly_data.data_utils import update_positions
                print("Triggering position refresh after stale trade cleanup...")
                update_positions(avgOnly=False)  # Full refresh to get accurate positions
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
                cleanup_inactive_markets()  # Clean up positions on removed markets
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
    logger = get_logger('main')
    global_state.logger = logger

    # Initialize client
    global_state.client = PolymarketClient()

    # Initialize state and fetch initial data
    global_state.all_tokens = []
    update_once()
    logger.info(f"After initial updates - Orders: {len(global_state.orders)}, Positions: {len(global_state.positions)}")

    # Clean up any positions/orders on markets that are no longer being traded
    cleanup_inactive_markets()

    logger.info("")
    logger.info(f'There are {len(global_state.df)} markets, {len(global_state.positions)} positions and {len(global_state.orders)} orders.')
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