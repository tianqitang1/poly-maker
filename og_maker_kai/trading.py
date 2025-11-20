"""
OG Maker Trading Logic - Original Strategy
Simpler, more aggressive market making from commit a6f87c5
"""
import gc                       # Garbage collection
import os                       # Operating system interface
import json                     # JSON handling
import asyncio                  # Asynchronous I/O
import traceback                # Exception handling
import pandas as pd             # Data analysis library
import math                     # Mathematical functions
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import poly_data.global_state as global_state
import poly_data.CONSTANTS as CONSTANTS

# Import utility functions for trading
from poly_data.trading_utils import get_best_bid_ask_deets, get_order_prices, get_buy_sell_amount, round_down, round_up
from poly_data.data_utils import get_position, get_order, set_position
from poly_utils.logging_utils import get_logger

# Initialize logger
logger = get_logger('og_maker_kai')

# Create directory for storing position risk information
if not os.path.exists('og_maker_kai/positions/'):
    os.makedirs('og_maker_kai/positions/')

def send_buy_order(order):
    """
    Create a BUY order for a specific token.

    This function:
    1. Cancels any existing orders for the token
    2. Checks if the order price is within acceptable range
    3. Creates a new buy order if conditions are met

    Args:
        order (dict): Order details including token, price, size, and market parameters
    """
    client = global_state.client

    # Only cancel existing orders if we need to make significant changes
    existing_buy_size = order['orders']['buy']['size']
    existing_buy_price = order['orders']['buy']['price']

    # Cancel orders if price changed significantly or size needs major adjustment
    price_diff = abs(existing_buy_price - order['price']) if existing_buy_price > 0 else float('inf')
    size_diff = abs(existing_buy_size - order['size']) if existing_buy_size > 0 else float('inf')

    should_cancel = (
        price_diff > 0.005 or  # Cancel if price diff > 0.5 cents
        size_diff > order['size'] * 0.1 or  # Cancel if size diff > 10%
        existing_buy_size == 0  # Cancel if no existing buy order
    )

    if should_cancel:
        logger.info(f"Cancelling buy orders - price diff: {price_diff:.4f}, size diff: {size_diff:.1f}")
        client.cancel_all_asset(order['token'])
    elif not should_cancel:
        logger.info(f"Keeping existing buy orders - minor changes: price diff: {price_diff:.4f}, size diff: {size_diff:.1f}")
        return  # Don't place new order if existing one is fine

    # Calculate minimum acceptable price based on market spread
    # Fix precision issue by rounding to appropriate decimal places
    round_length = 2  # Default fallback
    if 'row' in order:
        try:
            round_length = len(str(order['row']['tick_size']).split(".")[1])
        except:
            pass
            
    incentive_start = round(order['mid_price'] - order['max_spread']/100, round_length)
    
    # Add a tiny epsilon to avoid floating point comparison issues where 0.48 < 0.4800000000000004
    # But rounding above should handle it.
    
    trade = True

    # Don't place orders that are below incentive threshold
    if order['price'] < incentive_start:
        trade = False

    if trade:
        # Only place orders with prices between 0.1 and 0.9 to avoid extreme positions
        if order['price'] >= 0.1 and order['price'] < 0.9:
            try:
                client.create_order(
                    order['token'],
                    'BUY',
                    order['price'],
                    order['size'],
                    True if order['neg_risk'] == 'TRUE' else False,
                    raise_on_error=True
                )
            except Exception as ex:
                # Check if this is a balance/allowance error (not enough funds)
                error_msg = str(ex).lower()
                if 'balance' in error_msg or 'allowance' in error_msg:
                    logger.warning("WARNING: Buy order failed with balance/allowance error - not enough capital")
                else:
                    # Re-raise if it's a different error
                    raise
        else:
            logger.info("Not creating buy order because its outside acceptable price range (0.1-0.9)")
    else:
        logger.info(f'Not creating new order because order price of {order["price"]} is less than incentive start price of {incentive_start}. Mid price is {order["mid_price"]}')


def send_sell_order(order):
    """
    Create a SELL order for a specific token.

    This function:
    1. Cancels any existing orders for the token
    2. Creates a new sell order with the specified parameters

    Args:
        order (dict): Order details including token, price, size, and market parameters
    """
    client = global_state.client

    # Only cancel existing orders if we need to make significant changes
    existing_sell_size = order['orders']['sell']['size']
    existing_sell_price = order['orders']['sell']['price']

    # Cancel orders if price changed significantly or size needs major adjustment
    price_diff = abs(existing_sell_price - order['price']) if existing_sell_price > 0 else float('inf')
    size_diff = abs(existing_sell_size - order['size']) if existing_sell_size > 0 else float('inf')

    should_cancel = (
        price_diff > 0.005 or  # Cancel if price diff > 0.5 cents
        size_diff > order['size'] * 0.1 or  # Cancel if size diff > 10%
        existing_sell_size == 0  # Cancel if no existing sell order
    )

    if should_cancel:
        logger.info(f"Cancelling sell orders - price diff: {price_diff:.4f}, size diff: {size_diff:.1f}")
        client.cancel_all_asset(order['token'])
    elif not should_cancel:
        logger.info(f"Keeping existing sell orders - minor changes: price diff: {price_diff:.4f}, size diff: {size_diff:.1f}")
        return  # Don't place new order if existing one is fine

    try:
        client.create_order(
            order['token'],
            'SELL',
            order['price'],
            order['size'],
            True if order['neg_risk'] == 'TRUE' else False,
            raise_on_error=True
        )
    except Exception as ex:
        # Check if this is a balance/allowance error (position already sold / not enough balance)
        error_msg = str(ex).lower()
        if 'balance' in error_msg or 'allowance' in error_msg:
            logger.warning(
                f"WARNING: Sell order failed with balance/allowance error. "
                f"Refreshing cached position for token {order['token']}"
            )
            # Force update the actual position from blockchain and reconcile with cached state
            from poly_data.data_utils import get_position, set_position
            try:
                # On-chain size (in shares)
                actual_position = client.get_position(order['token'])[0] / 10**6

                # Cached size/avg from in-memory state
                cached_pos = get_position(order['token'])
                cached_size = float(cached_pos.get('size', 0))
                current_avg = float(cached_pos.get('avgPrice', 0))

                logger.info(
                    f"  Cached size {cached_size}, on-chain size {actual_position} "
                    f"for token {order['token']}"
                )

                # If there's a mismatch, adjust the cached position toward on-chain reality
                delta = cached_size - actual_position
                if abs(delta) > 0:
                    if delta > 0:
                        # We thought we had more than we actually do → reduce cached size
                        adj_side = 'SELL'
                        adj_size = delta
                    else:
                        # We thought we had less than we actually do → increase cached size
                        adj_side = 'BUY'
                        adj_size = -delta

                    set_position(order['token'], adj_side, adj_size, current_avg, 'correction')
                    logger.info(
                        f"  Adjusted cached position by {delta} shares "
                        f"(side={adj_side}), new cached size {get_position(order['token'])['size']}"
                    )
            except Exception as refresh_ex:
                logger.error(f"  Error reconciling position after failed sell: {refresh_ex}")
        else:
            # Re-raise if it's a different error
            raise

# Dictionary to store locks for each market to prevent concurrent trading on the same market
market_locks = {}

async def perform_trade(market):
    """
    Main trading function that handles market making for a specific market.
    """
    # Create a lock for this market if it doesn't exist
    if market not in market_locks:
        market_locks[market] = asyncio.Lock()

    # Use lock to prevent concurrent trading on the same market
    async with market_locks[market]:
        try:
            client = global_state.client
            # Get market details from the configuration
            if market not in global_state.df['condition_id'].values:
                return

            row = global_state.df[global_state.df['condition_id'] == market].iloc[0]
            # Determine decimal precision from tick size
            round_length = len(str(row['tick_size']).split(".")[1])

            # Get trading parameters for this market type
            params = global_state.params[row['param_type']]

            # Create a list with both outcomes for the market
            deets = [
                {'name': 'token1', 'token': row['token1'], 'answer': row['answer1']},
                {'name': 'token2', 'token': row['token2'], 'answer': row['answer2']}
            ]
            logger.info(f"\n\n{pd.Timestamp.utcnow().tz_localize(None)}: {row['question']}")
            
            # Use primary token (YES) for price lookups to ensure consistent inversion logic
            if market not in global_state.all_data:
                logger.info(f"Waiting for data on market {market}")
                return
            
            # Check data source orientation
            # If the stored book asset matches token2 (NO token), we need to swap our lookup logic
            stored_asset_id = str(global_state.all_data[market].get('asset_id', ''))
            is_book_inverted = (stored_asset_id == str(row['token2']))
            global_state.all_data[market]['book_inverted'] = is_book_inverted
            
            if is_book_inverted:
                 logger.debug(f"Market {market}: Book data is for NO token. adjusting price lookups.")

            # Get current positions for both outcomes
            pos_1 = get_position(row['token1'])['size']
            pos_2 = get_position(row['token2'])['size']

            # ------- POSITION MERGING LOGIC -------
            # Calculate if we have opposing positions that can be merged
            amount_to_merge = min(pos_1, pos_2)

            # Only merge if positions are above minimum threshold
            if float(amount_to_merge) > CONSTANTS.MIN_MERGE_SIZE:
                # Get exact position sizes from blockchain for merging
                pos_1 = client.get_position(row['token1'])[0]
                pos_2 = client.get_position(row['token2'])[0]
                amount_to_merge = min(pos_1, pos_2)
                scaled_amt = amount_to_merge / 10**6

                if scaled_amt > CONSTANTS.MIN_MERGE_SIZE:
                    logger.info(f"Position 1 is of size {pos_1} and Position 2 is of size {pos_2}. Merging positions")
                    # Execute the merge operation
                    client.merge_positions(amount_to_merge, market, row['neg_risk'] == 'TRUE')
                    # Update our local position tracking
                    set_position(row['token1'], 'SELL', scaled_amt, 0, 'merge')
                    set_position(row['token2'], 'SELL', scaled_amt, 0, 'merge')

            # Store potential buy orders to execute them only if both sides are valid
            buy_candidates = []
            market_valid_for_buys = True
            failed_reason = None

            # ------- TRADING LOGIC FOR EACH OUTCOME -------
            # Loop through both outcomes in the market (YES and NO)
            for detail in deets:
                token = int(detail['token'])

                # Get current orders for this token
                orders = get_order(token)

                # Determine lookup name for pricing utils
                # Standard: token1 (YES) book -> 'token1' reads direct, 'token2' inverts
                # Inverted: token2 (NO) book  -> 'token1' needs to invert (ask for 'token2'), 'token2' reads direct (ask for 'token1')
                lookup_name = detail['name']
                if is_book_inverted:
                    lookup_name = 'token2' if detail['name'] == 'token1' else 'token1'

                # Get market depth and price information
                # Use min_size from config to determine what counts as "real" liquidity
                # This allows handling both thin/volatile markets (low min_size) and stable ones (high min_size)
                check_size = max(row.get('min_size', 5), 5)
                deets_data = get_best_bid_ask_deets(market, lookup_name, check_size, 0.1)

                #if deet has None for one these values below, call it with min size of 5
                if deets_data['best_bid'] is None or deets_data['best_ask'] is None or deets_data['best_bid_size'] is None or deets_data['best_ask_size'] is None:
                    deets_data = get_best_bid_ask_deets(market, lookup_name, 5, 0.1)

                # Detect collapsed liquidity by rechecking with a smaller request size
                fallback_size = max(5, min(check_size / 2, check_size))
                best_bid = deets_data['best_bid']
                best_bid_size = deets_data['best_bid_size']
                best_ask = deets_data['best_ask']
                best_ask_size = deets_data['best_ask_size']
                need_refine = False
                observed_spread = None

                if best_bid is None or best_ask is None:
                    need_refine = True
                elif check_size > 5:
                    observed_spread = best_ask - best_bid
                    tight_threshold = max(params['spread_threshold'] * 2, 0.05)
                    if observed_spread is not None and observed_spread > tight_threshold:
                        need_refine = True
                    depth_floor = check_size * 0.5
                    if (best_bid_size is not None and best_bid_size < depth_floor) or (best_ask_size is not None and best_ask_size < depth_floor):
                        need_refine = True

                if need_refine and fallback_size < check_size:
                    refined = get_best_bid_ask_deets(market, lookup_name, fallback_size, 0.1)
                    if refined['best_bid'] is not None and refined['best_ask'] is not None:
                        logger.info(f"Detected thin book for {detail['answer']} – falling back to size {fallback_size} from {check_size}")
                        deets_data = refined
                        check_size = fallback_size

                # Extract all order book details after any refinement
                best_bid = deets_data['best_bid']
                best_bid_size = deets_data['best_bid_size']
                second_best_bid = deets_data['second_best_bid']
                second_best_bid_size = deets_data['second_best_bid_size']
                top_bid = deets_data['top_bid']
                best_ask = deets_data['best_ask']
                best_ask_size = deets_data['best_ask_size']
                second_best_ask = deets_data['second_best_ask']
                second_best_ask_size = deets_data['second_best_ask_size']
                top_ask = deets_data['top_ask']

                # Patch book with our local orders to prevent race conditions/lag
                my_buy_price = orders['buy']['price']
                my_buy_size = orders['buy']['size']
                if my_buy_size > 0:
                    if best_bid is None or my_buy_price >= best_bid:
                         best_bid = my_buy_price
                         best_bid_size = my_buy_size 
                    if top_bid is None or my_buy_price > top_bid:
                             top_bid = my_buy_price

                my_sell_price = orders['sell']['price']
                my_sell_size = orders['sell']['size']
                if my_sell_size > 0:
                    if best_ask is None or my_sell_price <= best_ask:
                         best_ask = my_sell_price
                         best_ask_size = my_sell_size
                    if top_ask is None or my_sell_price < top_ask:
                             top_ask = my_sell_price

                # Round prices to appropriate precision
                if best_bid is not None: best_bid = round(best_bid, round_length)
                if best_ask is not None: best_ask = round(best_ask, round_length)
                if top_bid is not None: top_bid = round(top_bid, round_length)
                if top_ask is not None: top_ask = round(top_ask, round_length)

                # Calculate ratio of buy vs sell liquidity in the market
                try:
                    overall_ratio = (deets_data['bid_sum_within_n_percent']) / (deets_data['ask_sum_within_n_percent'])
                except:
                    overall_ratio = 0

                try:
                    second_best_bid = round(second_best_bid, round_length)
                    second_best_ask = round(second_best_ask, round_length)
                except:
                    pass

                top_bid = round(top_bid, round_length)
                top_ask = round(top_ask, round_length)

                # Get our current position and average price
                pos = get_position(token)
                position = pos['size']
                avgPrice = pos['avgPrice']

                position = round_down(position, 2)

                # Calculate optimal bid and ask prices based on market conditions
                bid_price, ask_price = get_order_prices(
                    best_bid, best_bid_size, top_bid, best_ask,
                    best_ask_size, top_ask, avgPrice, row
                )

                bid_price = round(bid_price, round_length)
                ask_price = round(ask_price, round_length)

                # Calculate mid price for reference
                mid_price = (top_bid + top_ask) / 2

                # Log market conditions for this outcome
                logger.info(f"\nFor {detail['answer']}. Orders: {orders} Position: {position}, "
                      f"avgPrice: {avgPrice}, Best Bid: {best_bid}, Best Ask: {best_ask}, "
                      f"Bid Price: {bid_price}, Ask Price: {ask_price}, Mid Price: {mid_price}")

                # Get position for the opposite token to calculate total exposure
                other_token = global_state.REVERSE_TOKENS[str(token)]
                other_position = get_position(other_token)['size']

                # Calculate how much to buy or sell based on our position
                buy_amount, sell_amount = get_buy_sell_amount(position, bid_price, row, other_position)

                # Get max_size for logging (same logic as in get_buy_sell_amount)
                max_size = row.get('max_size', row['trade_size'])

                # Prepare order object with all necessary information
                order = {
                    "token": token,
                    "mid_price": mid_price,
                    "neg_risk": row['neg_risk'],
                    "max_spread": row['max_spread'],
                    'orders': orders,
                    'token_name': detail['name'],
                    'row': row
                }

                logger.info(f"Position: {position}, Other Position: {other_position}, "
                      f"Trade Size: {row['trade_size']}, Max Size: {max_size}, "
                      f"buy_amount: {buy_amount}, sell_amount: {sell_amount}")

                # File to store risk management information for this market
                fname = 'og_maker_kai/positions/' + str(market) + '.json'

                # ------- SELL ORDER LOGIC (STOP-LOSS) ------- 
                if sell_amount > 0:
                    # Skip if we have no average price (no real position)
                    if avgPrice == 0:
                        logger.info("Avg Price is 0. Skipping")
                        # Continue to buy logic? No, sell logic is independent but loop continues
                    else:
                        order['size'] = sell_amount
                        order['price'] = ask_price

                        # Determine lookup name for risk assessment (same logic as above)
                        lookup_name = detail['name']
                        if is_book_inverted:
                            lookup_name = 'token2' if detail['name'] == 'token1' else 'token1'

                        # Get fresh market data for risk assessment with dynamic sizing
                        risk_check_size = max(check_size, row.get('min_size', 5), 5)
                        n_deets = get_best_bid_ask_deets(market, lookup_name, risk_check_size, 0.1)

                        if n_deets['best_bid'] is None or n_deets['best_ask'] is None:
                            n_deets = get_best_bid_ask_deets(market, lookup_name, 5, 0.1)

                        if n_deets['best_bid'] is None or n_deets['best_ask'] is None:
                            logger.warning(f"Unable to compute stop-loss prices for {detail['answer']} due to empty book")
                            continue

                        # Calculate current market price and spread
                        mid_price = round_up((n_deets['best_bid'] + n_deets['best_ask']) / 2, round_length)
                        spread = round(n_deets['best_ask'] - n_deets['best_bid'], 2)

                        # Calculate current profit/loss on position
                        pnl = (mid_price - avgPrice) / avgPrice * 100

                        logger.info(f"Mid Price: {mid_price}, Spread: {spread}, PnL: {pnl}")

                        # Prepare risk details for tracking
                        risk_details = {
                            'time': str(pd.Timestamp.utcnow().tz_localize(None)),
                            'question': row['question']
                        }

                        try:
                            ratio = (n_deets['bid_sum_within_n_percent']) / (n_deets['ask_sum_within_n_percent'])
                        except:
                            ratio = 0

                        # ORIGINAL: Sell only trade_size amount (partial exit)
                        pos_to_sell = sell_amount

                        # ------- STOP-LOSS LOGIC ------- 
                        # Trigger stop-loss if either:
                        # 1. PnL is below threshold (even if book is thin after double-checking)
                        # 2. Volatility is too high
                        stoploss_due_to_pnl = pnl < params['stop_loss_threshold']
                        spread_ok = spread <= params['spread_threshold']
                        liquidity_collapse = False

                        if stoploss_due_to_pnl and not spread_ok:
                            fallback_size = max(5, risk_check_size / 2)
                            refined = None
                            if fallback_size < risk_check_size:
                                refined = get_best_bid_ask_deets(market, lookup_name, fallback_size, 0.1)
                            if refined and refined['best_bid'] is not None and refined['best_ask'] is not None:
                                refined_spread = round(refined['best_ask'] - refined['best_bid'], 2)
                                if refined_spread <= params['spread_threshold']:
                                    n_deets = refined
                                    spread = refined_spread
                                    spread_ok = True
                                    mid_price = round_up((n_deets['best_bid'] + n_deets['best_ask']) / 2, round_length)
                                else:
                                    liquidity_collapse = True
                                    n_deets = refined
                            else:
                                liquidity_collapse = True

                        if (stoploss_due_to_pnl and (spread_ok or liquidity_collapse)) or row['3_hour'] > params['volatility_threshold']:
                            risk_details['msg'] = (f"Selling {pos_to_sell} because spread is {spread} and pnl is {pnl} "
                                                  f"and ratio is {ratio} and 3 hour volatility is {row['3_hour']}")
                            logger.info(f"Stop loss Triggered: {risk_details['msg']}")

                            if n_deets['best_bid'] is None:
                                logger.warning("Cannot execute stop loss – no best bid in fallback book")
                                continue

                            # Sell at market best bid to ensure execution
                            order['size'] = pos_to_sell
                            order['price'] = n_deets['best_bid']

                            # Set period to avoid trading after stop-loss
                            risk_details['sleep_till'] = str(pd.Timestamp.utcnow().tz_localize(None) +
                                                            pd.Timedelta(hours=params['sleep_period']))

                            logger.info("Risking off")
                            send_sell_order(order)
                            client.cancel_all_market(market)

                            # Save risk details to file
                            open(fname, 'w').write(json.dumps(risk_details))
                            
                            # If we trigger stop-loss, we likely shouldn't be buying either
                            market_valid_for_buys = False
                            failed_reason = "Stop loss triggered"
                            continue
                
                # ------- BUY ORDER LOGIC PREPARATION ------- 
                # Don't process buys if we already failed market validation
                if not market_valid_for_buys:
                    continue
                    
                # Get max_size, defaulting to trade_size if not specified
                max_size = row.get('max_size', row['trade_size'])

                # Check basic conditions
                if position < max_size and position < 250 and buy_amount > 0 and buy_amount >= row['min_size']:
                    # Get reference price from market data
                    sheet_value = row['best_bid']
                    if detail['name'] == 'token2':
                        sheet_value = 1 - row['best_ask']
                    sheet_value = round(sheet_value, round_length)
                    
                    order['size'] = buy_amount
                    order['price'] = bid_price

                    # Check if price is far from reference
                    price_change = abs(order['price'] - sheet_value)

                    # Validate Conditions
                    # 1. Risk-off period check
                    if os.path.isfile(fname):
                        risk_details = json.load(open(fname))
                        start_trading_at = pd.to_datetime(risk_details['sleep_till'])
                        current_time = pd.Timestamp.utcnow().tz_localize(None)

                        if current_time < start_trading_at:
                            logger.info(f"Skipping buy: Risked off until {start_trading_at}")
                            market_valid_for_buys = False
                            failed_reason = "Recently risked off"
                            continue

                    # 2. Volatility check
                    if row['3_hour'] > params['volatility_threshold']:
                        logger.info(f"Skipping buy for {detail['answer']}: Volatility {row['3_hour']} > {params['volatility_threshold']}")
                        continue
                        
                    # 3. Price deviation check
                    if price_change >= 0.05:
                        logger.info(f"Skipping buy for {detail['answer']}: Price deviation {price_change:.3f} >= 0.05")
                        continue
                        
                    # 4. Ratio check
                    if overall_ratio < 0:
                        logger.info(f"Skipping buy for {detail['answer']}: Ratio {overall_ratio} < 0")
                        continue

                    # 5. Incentive Start Price Check (The specific error user saw)
                    incentive_start = round(mid_price - row['max_spread']/100, round_length)
                    if order['price'] < incentive_start:
                         logger.info(f"Skipping buy for {detail['answer']}: Price {order['price']} < Incentive Start {incentive_start}")
                         continue

                    # 6. Determine if we *need* to send an order
                    should_send = False
                    if best_bid > orders['buy']['price']:
                        should_send = True
                    elif position + orders['buy']['size'] < 0.95 * max_size:
                        should_send = True
                    elif orders['buy']['size'] > order['size'] * 1.01:
                        should_send = True
                    
                    if should_send:
                        buy_candidates.append(order)

                # ------- TAKE PROFIT / SELL ORDER MANAGEMENT ------- 
                # This is managing existing position exits, should run independently of buy-side validation
                elif sell_amount > 0:
                    order['size'] = sell_amount

                    # OG MAKER KAI: Volatility-Adjusted Take-Profit
                    base_tp_threshold = params['take_profit_threshold']
                    volatility_factor = 1.0
                    if row['3_hour'] > params['volatility_threshold'] * 0.5:
                        vol_ratio = min(row['3_hour'] / params['volatility_threshold'], 1.5)
                        volatility_factor = max(1.0, vol_ratio)
                        
                    adjusted_tp_threshold = base_tp_threshold * volatility_factor
                    
                    tp_price = round_up(avgPrice + (avgPrice * adjusted_tp_threshold/100), round_length)
                    order['price'] = round_up(tp_price if ask_price < tp_price else ask_price, round_length)

                    tp_price = float(tp_price)
                    order_price = float(orders['sell']['price'])

                    diff = abs(order_price - tp_price)/tp_price * 100

                    if diff > 2:
                        logger.info(f"Sending Sell Order for {token} (TP update) - diff {diff:.2f}%")
                        send_sell_order(order)
                    elif orders['sell']['size'] < position * 0.97:
                        logger.info(f"Sending Sell Order for {token} (TP size update)")
                        send_sell_order(order)

            # ------- EXECUTE BUYS IF VALID ------- 
            if not market_valid_for_buys:
                logger.info(f"Skipping ALL buy orders for {row['question']} due to: {failed_reason}")
                # Optionally cancel all buys if market is invalid?
                # client.cancel_all_market(market) # Maybe too aggressive? 
                # For now, just don't place new ones.
            else:
                # If we are here, both sides PASSED the validity checks (or didn't fail them).
                # However, we might only have 1 candidate if the other side didn't need an order update (already optimal).
                # That is fine. The requirement is "do not place IF requirements for OTHER side are NOT MET".
                # Since market_valid_for_buys is True, requirements ARE met for both sides.
                
                if len(buy_candidates) > 0:
                    logger.info(f"Executing {len(buy_candidates)} valid buy orders")
                    for order in buy_candidates:
                        send_buy_order(order)

        except Exception as ex:
            logger.error(f"Error performing trade for {market}: {ex}", exc_info=True)

        # Clean up memory and introduce a small delay
        gc.collect()
        await asyncio.sleep(2)
