import gc                       # Garbage collection
import os                       # Operating system interface
import json                     # JSON handling
import asyncio                  # Asynchronous I/O
import traceback                # Exception handling
import pandas as pd             # Data analysis library
import numpy as np              # Numerical Python
import math                     # Mathematical functions

import poly_data.global_state as global_state
import poly_data.CONSTANTS as CONSTANTS

# Import utility functions for trading
from poly_data.trading_utils import get_best_bid_ask_deets, get_order_prices, get_buy_sell_amount, round_down, round_up
from poly_data.data_utils import get_position, get_order, set_position

# Create directory for storing position risk information
if not os.path.exists('positions/'):
    os.makedirs('positions/')

def convert_to_serializable(obj):
    """
    Convert numpy/pandas types to native Python types for JSON serialization.

    Args:
        obj: Any object that might contain numpy/pandas types

    Returns:
        Object with all numpy/pandas types converted to native Python types
    """
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # pandas types
        return obj.item()
    else:
        return obj

def send_buy_order(order, logger=None):
    """
    Create a BUY order for a specific token.

    This function:
    1. Cancels any existing orders for the token
    2. Checks if the order price is within acceptable range
    3. Creates a new buy order if conditions are met

    Args:
        order (dict): Order details including token, price, size, and market parameters
        logger: BotLogger instance (optional)
    """
    # Get logger from global state if not provided
    if logger is None and hasattr(global_state, 'logger'):
        logger = global_state.logger

    client = global_state.client

    # Safety check: get current position and ensure order won't exceed max_size
    from poly_data.data_utils import get_position
    current_pos = get_position(order['token'])['size']
    max_size = order['row'].get('max_size', order['row']['trade_size'])

    # If current position + order size would exceed max_size * 1.1, reduce order size
    if current_pos + order['size'] > max_size * 1.1:
        msg = f"WARNING: Position ({current_pos}) + Order ({order['size']}) would exceed max_size ({max_size}) by >10%"
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        # Cancel all orders to prevent over-sizing
        client.cancel_all_asset(order['token'])
        return

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
    
    if should_cancel and (existing_buy_size > 0 or order['orders']['sell']['size'] > 0):
        msg = f"Cancelling buy orders - price diff: {price_diff:.4f}, size diff: {size_diff:.1f}"
        if logger:
            logger.debug(msg)
        else:
            print(msg)
        client.cancel_all_asset(order['token'])
    elif not should_cancel:
        msg = f"Keeping existing buy orders - minor changes: price diff: {price_diff:.4f}, size diff: {size_diff:.1f}"
        if logger:
            logger.debug(msg)
        else:
            print(msg)
        return  # Don't place new order if existing one is fine

    # Calculate minimum acceptable price based on market spread
    incentive_start = order['mid_price'] - order['max_spread']/100

    trade = True

    # Don't place orders that are below incentive threshold
    if order['price'] < incentive_start:
        trade = False

    if trade:
        # Only place orders with prices between 0.1 and 0.9 to avoid extreme positions
        if order['price'] >= 0.1 and order['price'] < 0.9:
            if logger:
                logger.info(f'Creating BUY order for {order["size"]} at {order["price"]}')
            else:
                print(f'Creating new order for {order["size"]} at {order["price"]}')
                print(order['token'], 'BUY', order['price'], order['size'])
            try:
                response = client.create_order(
                    order['token'],
                    'BUY',
                    order['price'],
                    order['size'],
                    True if order['neg_risk'] == 'TRUE' else False
                )

                # Log order placement
                if logger and response:
                    logger.log_order(
                        action='BUY',
                        market=order['row']['question'],
                        token=str(order['token']),
                        price=order['price'],
                        size=order['size'],
                        order_id=response.get('orderID', 'N/A') if isinstance(response, dict) else 'N/A'
                    )
            except Exception as ex:
                # Check if this is a balance/allowance error (not enough funds)
                error_msg = str(ex).lower()
                if 'balance' in error_msg or 'allowance' in error_msg:
                    msg = "WARNING: Buy order failed with balance/allowance error - not enough capital"
                    if logger:
                        logger.warning(msg)
                    else:
                        print(msg)
                else:
                    # Re-raise if it's a different error
                    raise
        else:
            msg = "Not creating buy order because its outside acceptable price range (0.1-0.9)"
            if logger:
                logger.debug(msg)
            else:
                print(msg)
    else:
        msg = f'Not creating new order because order price of {order["price"]} is less than incentive start price of {incentive_start}. Mid price is {order["mid_price"]}'
        if logger:
            logger.debug(msg)
        else:
            print(msg)


def send_sell_order(order, logger=None):
    """
    Create a SELL order for a specific token.

    This function:
    1. Cancels any existing orders for the token
    2. Creates a new sell order with the specified parameters

    Args:
        order (dict): Order details including token, price, size, and market parameters
        logger: BotLogger instance (optional)
    """
    # Get logger from global state if not provided
    if logger is None and hasattr(global_state, 'logger'):
        logger = global_state.logger

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
    
    if should_cancel and (existing_sell_size > 0 or order['orders']['buy']['size'] > 0):
        msg = f"Cancelling sell orders - price diff: {price_diff:.4f}, size diff: {size_diff:.1f}"
        if logger:
            logger.debug(msg)
        else:
            print(msg)
        client.cancel_all_asset(order['token'])
    elif not should_cancel:
        msg = f"Keeping existing sell orders - minor changes: price diff: {price_diff:.4f}, size diff: {size_diff:.1f}"
        if logger:
            logger.debug(msg)
        else:
            print(msg)
        return  # Don't place new order if existing one is fine

    if logger:
        logger.info(f'Creating SELL order for {order["size"]} at {order["price"]}')
    else:
        print(f'Creating new order for {order["size"]} at {order["price"]}')
    try:
        response = client.create_order(
            order['token'],
            'SELL',
            order['price'],
            order['size'],
            True if order['neg_risk'] == 'TRUE' else False
        )

        # Log order placement
        if logger and response:
            logger.log_order(
                action='SELL',
                market=order['row']['question'],
                token=str(order['token']),
                price=order['price'],
                size=order['size'],
                order_id=response.get('orderID', 'N/A') if isinstance(response, dict) else 'N/A'
            )
    except Exception as ex:
        # Check if this is a balance/allowance error (position already sold)
        error_msg = str(ex).lower()
        if 'balance' in error_msg or 'allowance' in error_msg:
            msg = f"WARNING: Sell order failed with balance/allowance error. Refreshing position for token {order['token']}"
            if logger:
                logger.warning(msg)
            else:
                print(msg)
            # Force update the actual position from blockchain
            from poly_data.data_utils import get_position, set_position
            try:
                actual_position = client.get_position(order['token'])[0] / 10**6
                current_avg = get_position(order['token'])['avgPrice']
                msg = f"  Cached position was {order['size']}, actual on-chain position is {actual_position}"
                if logger:
                    logger.info(msg)
                else:
                    print(msg)
                # Update our cached position to match reality
                if actual_position != order['size']:
                    set_position(order['token'], 'SELL', order['size'] - actual_position, current_avg, 'correction')
                    msg = f"  Updated cached position to {actual_position}"
                    if logger:
                        logger.info(msg)
                    else:
                        print(msg)
            except Exception as refresh_ex:
                msg = f"  Error refreshing position: {refresh_ex}"
                if logger:
                    logger.error(msg)
                else:
                    print(msg)
        else:
            # Re-raise if it's a different error
            raise

# Dictionary to store locks for each market to prevent concurrent trading on the same market
market_locks = {}

async def perform_trade(market, logger=None):
    """
    Main trading function that handles market making for a specific market.

    This function:
    1. Merges positions when possible to free up capital
    2. Analyzes the market to determine optimal bid/ask prices
    3. Manages buy and sell orders based on position size and market conditions
    4. Implements risk management with stop-loss and take-profit logic

    Args:
        market (str): The market ID to trade on
        logger: BotLogger instance (optional, will use global_state.logger if available)
    """
    # Get logger from global state if not provided
    if logger is None and hasattr(global_state, 'logger'):
        logger = global_state.logger

    # Create a lock for this market if it doesn't exist
    if market not in market_locks:
        market_locks[market] = asyncio.Lock()

    # Use lock to prevent concurrent trading on the same market
    async with market_locks[market]:
        try:
            client = global_state.client
            # Get market details from the configuration
            market_df = global_state.df[global_state.df['condition_id'] == market]

            # Check if market exists in configuration
            if market_df.empty:
                if logger:
                    logger.warning(f"Market {market} not found in configuration dataframe. Skipping trade.")
                else:
                    print(f"Warning: Market {market} not found in configuration dataframe. Skipping trade.")
                return

            row = market_df.iloc[0]
            # Determine decimal precision from tick size
            round_length = len(str(row['tick_size']).split(".")[1])

            # Get trading parameters for this market type
            params = global_state.params[row['param_type']]

            # Create a list with both outcomes for the market
            deets = [
                {'name': 'token1', 'token': row['token1'], 'answer': row['answer1']},
                {'name': 'token2', 'token': row['token2'], 'answer': row['answer2']}
            ]
            if logger:
                logger.info(f"\n{pd.Timestamp.utcnow().tz_localize(None)}: {row['question']}")
            else:
                print(f"\n\n{pd.Timestamp.utcnow().tz_localize(None)}: {row['question']}")

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
                    if logger:
                        logger.info(f"Position 1 is of size {pos_1} and Position 2 is of size {pos_2}. Merging positions")
                    else:
                        print(f"Position 1 is of size {pos_1} and Position 2 is of size {pos_2}. Merging positions")

                    # Execute the merge operation (works with Gnosis Safe wallets)
                    client.merge_positions(amount_to_merge, market, row['neg_risk'] == 'TRUE')

                    # Update our local position tracking
                    set_position(row['token1'], 'SELL', scaled_amt, 0, 'merge')
                    set_position(row['token2'], 'SELL', scaled_amt, 0, 'merge')

                    # Log merge
                    if logger:
                        logger.log_merge(
                            market=row['question'],
                            size=scaled_amt,
                            recovered=scaled_amt,
                            condition_id=market
                        )
                    
            # ------- TRADING LOGIC FOR EACH OUTCOME -------
            # Loop through both outcomes in the market (YES and NO)
            for detail in deets:
                token = int(detail['token'])
                
                # Get current orders for this token
                orders = get_order(token)

                # Get market depth and price information
                deets = get_best_bid_ask_deets(market, detail['name'], 100, 0.1)

                #if deet has None for one these values below, call it with min size of 20
                if deets['best_bid'] is None or deets['best_ask'] is None or deets['best_bid_size'] is None or deets['best_ask_size'] is None:
                    deets = get_best_bid_ask_deets(market, detail['name'], 20, 0.1)

                # Extract all order book details
                best_bid = deets['best_bid']
                best_bid_size = deets['best_bid_size']
                second_best_bid = deets['second_best_bid']
                second_best_bid_size = deets['second_best_bid_size']
                top_bid = deets['top_bid']
                best_ask = deets['best_ask']
                best_ask_size = deets['best_ask_size']
                second_best_ask = deets['second_best_ask']
                second_best_ask_size = deets['second_best_ask_size']
                top_ask = deets['top_ask']

                # Check if we still have None values after retry - skip if market has insufficient data
                if best_bid is None or best_ask is None or top_bid is None or top_ask is None:
                    msg = f"Skipping {detail['answer']}: Insufficient market data (best_bid={best_bid}, best_ask={best_ask}, top_bid={top_bid}, top_ask={top_ask})"
                    if logger:
                        logger.warning(msg)
                    else:
                        print(msg)
                    continue

                # Round prices to appropriate precision
                best_bid = round(best_bid, round_length)
                best_ask = round(best_ask, round_length)

                # Calculate ratio of buy vs sell liquidity in the market
                try:
                    overall_ratio = (deets['bid_sum_within_n_percent']) / (deets['ask_sum_within_n_percent'])
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
                msg = (f"For {detail['answer']}. Orders: {orders} Position: {position}, "
                       f"avgPrice: {avgPrice}, Best Bid: {best_bid}, Best Ask: {best_ask}, "
                       f"Bid Price: {bid_price}, Ask Price: {ask_price}, Mid Price: {mid_price}")
                if logger:
                    logger.debug(msg)
                else:
                    print(f"\n{msg}")

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
            
                msg = (f"Position: {position}, Other Position: {other_position}, "
                       f"Trade Size: {row['trade_size']}, Max Size: {max_size}, "
                       f"buy_amount: {buy_amount}, sell_amount: {sell_amount}")
                if logger:
                    logger.debug(msg)
                else:
                    print(msg)

                # File to store risk management information for this market
                fname = 'positions/' + str(market) + '.json'

                # ------- SELL ORDER LOGIC -------
                if sell_amount > 0:
                    # Skip if we have no average price (no real position)
                    if avgPrice == 0:
                        if logger:
                            logger.debug("Avg Price is 0. Skipping")
                        else:
                            print("Avg Price is 0. Skipping")
                        continue

                    order['size'] = sell_amount
                    order['price'] = ask_price

                    # Get fresh market data for risk assessment
                    n_deets = get_best_bid_ask_deets(market, detail['name'], 100, 0.1)

                    # Retry with smaller size if needed
                    if n_deets['best_bid'] is None or n_deets['best_ask'] is None:
                        n_deets = get_best_bid_ask_deets(market, detail['name'], 20, 0.1)

                    # Check if we still have None values - skip sell logic if market data unavailable
                    if n_deets['best_bid'] is None or n_deets['best_ask'] is None:
                        msg = f"Skipping sell logic for {detail['answer']}: Insufficient market data (best_bid={n_deets['best_bid']}, best_ask={n_deets['best_ask']})"
                        if logger:
                            logger.warning(msg)
                        else:
                            print(msg)
                        continue

                    # Calculate current market price and spread
                    mid_price = round_up((n_deets['best_bid'] + n_deets['best_ask']) / 2, round_length)
                    spread = round(n_deets['best_ask'] - n_deets['best_bid'], 2)

                    # Calculate current profit/loss on position
                    pnl = (mid_price - avgPrice) / avgPrice * 100

                    msg = f"Mid Price: {mid_price}, Spread: {spread}, PnL: {pnl}"
                    if logger:
                        logger.debug(msg)
                    else:
                        print(msg)
                    
                    # Prepare risk details for tracking
                    risk_details = {
                        'time': str(pd.Timestamp.utcnow().tz_localize(None)),
                        'question': row['question']
                    }

                    try:
                        ratio = (n_deets['bid_sum_within_n_percent']) / (n_deets['ask_sum_within_n_percent'])
                    except:
                        ratio = 0

                    # When stop-loss triggers, sell the ENTIRE position to exit quickly
                    pos_to_sell = position

                    # ------- STOP-LOSS LOGIC -------
                    # Trigger stop-loss if either:
                    # 1. PnL is below threshold and spread is tight enough to exit
                    # 2. Volatility is too high
                    if (pnl < params['stop_loss_threshold'] and spread <= params['spread_threshold']) or row['3_hour'] > params['volatility_threshold']:
                        risk_details['msg'] = (f"Selling {pos_to_sell} because spread is {spread} and pnl is {pnl} "
                                              f"and ratio is {ratio} and 3 hour volatility is {row['3_hour']}")

                        if logger:
                            logger.warning(f"Stop loss Triggered: {risk_details['msg']}")
                        else:
                            print("Stop loss Triggered: ", risk_details['msg'])

                        # Sell at market best bid to ensure execution
                        order['size'] = pos_to_sell
                        order['price'] = n_deets['best_bid']

                        # Set period to avoid trading after stop-loss
                        risk_details['sleep_till'] = str(pd.Timestamp.utcnow().tz_localize(None) +
                                                        pd.Timedelta(hours=params['sleep_period']))

                        # Calculate PnL in dollars
                        pnl_dollars = (mid_price - avgPrice) * pos_to_sell

                        # Log stop loss
                        if logger:
                            logger.log_stop_loss(
                                market=row['question'],
                                position=pos_to_sell,
                                entry_price=avgPrice,
                                exit_price=n_deets['best_bid'],
                                pnl=pnl_dollars,
                                reason=f"PnL: {pnl:.2f}%, Spread: {spread:.4f}, Volatility: {row['3_hour']:.2f}"
                            )

                        if logger:
                            logger.info("Risking off - selling position and cancelling all orders")
                        else:
                            print("Risking off")

                        send_sell_order(order, logger)
                        client.cancel_all_market(market)

                        # Save risk details to file
                        # Convert numpy/pandas types to native Python types for JSON serialization
                        serializable_risk_details = convert_to_serializable(risk_details)
                        open(fname, 'w').write(json.dumps(serializable_risk_details))
                        continue

                # ------- BUY ORDER LOGIC -------
                # Get max_size, defaulting to trade_size if not specified
                max_size = row.get('max_size', row['trade_size'])

                # Cancel buy orders if position is at or above 90% of max_size to prevent over-sizing
                if position >= max_size * 0.9 and orders['buy']['size'] > 0:
                    msg = f"Cancelling buy orders - position ({position}) at or above 90% of max_size ({max_size})"
                    if logger:
                        logger.info(msg)
                    else:
                        print(msg)
                    client.cancel_all_asset(token)
                    # Set buy_amount to 0 to skip buy logic below
                    buy_amount = 0

                # Only buy if:
                # 1. Position is less than max_size (new logic)
                # 2. Position is less than absolute cap (250)
                # 3. Buy amount is above minimum size
                if position < max_size and position < 250 and buy_amount > 0 and buy_amount >= row['min_size']:
                    # EARLY CHECK: Don't buy if we have opposing position
                    rev_token = global_state.REVERSE_TOKENS[str(token)]
                    rev_pos = get_position(rev_token)

                    # Two thresholds:
                    # 1. If reverse position is >= 80% of max_size, definitely don't buy opposite side
                    # 2. If reverse position > min_size or > 10% of max_size, don't buy
                    if rev_pos['size'] >= max_size * 0.8 or rev_pos['size'] > max(row['min_size'], max_size * 0.1):
                        msg = f"Bypassing buy logic for {detail['answer']} - opposing position of {rev_pos['size']} exists (max_size: {max_size}, min_size: {row['min_size']})"
                        if logger:
                            logger.debug(msg)
                        else:
                            print(msg)
                        if orders['buy']['size'] > CONSTANTS.MIN_MERGE_SIZE:
                            msg2 = f"Cancelling buy orders because there is a reverse position of {rev_pos['size']}"
                            if logger:
                                logger.debug(msg2)
                            else:
                                print(msg2)
                            client.cancel_all_asset(token)
                        continue

                    # Get reference price from market data
                    sheet_value = row['best_bid']

                    if detail['name'] == 'token2':
                        sheet_value = 1 - row['best_ask']

                    sheet_value = round(sheet_value, round_length)
                    order['size'] = buy_amount
                    order['price'] = bid_price

                    # Check if price is far from reference
                    price_change = abs(order['price'] - sheet_value)

                    send_buy = True

                    # ------- RISK-OFF PERIOD CHECK -------
                    # If we're in a risk-off period (after stop-loss), don't buy
                    if os.path.isfile(fname):
                        risk_details = json.load(open(fname))

                        start_trading_at = pd.to_datetime(risk_details['sleep_till'])
                        current_time = pd.Timestamp.utcnow().tz_localize(None)

                        if logger:
                            logger.debug(f"Risk details: {risk_details}, current_time: {current_time}, start_trading_at: {start_trading_at}")
                        else:
                            print(risk_details, current_time, start_trading_at)
                        if current_time < start_trading_at:
                            send_buy = False
                            msg = f"Not sending a buy order because recently risked off. Risked off at {risk_details['time']}"
                            if logger:
                                logger.info(msg)
                            else:
                                print(msg)

                    # Only proceed if we're not in risk-off period
                    if send_buy:
                        # Don't buy if volatility is high or price is far from reference
                        if row['3_hour'] > params['volatility_threshold'] or price_change >= 0.05:
                            msg = (f'3 Hour Volatility of {row["3_hour"]} is greater than max volatility of '
                                  f'{params["volatility_threshold"]} or price of {order["price"]} is outside '
                                  f'0.05 of {sheet_value}. Cancelling all orders')
                            if logger:
                                logger.info(msg)
                            else:
                                print(msg)
                            client.cancel_all_asset(order['token'])
                        else:
                            # Check market buy/sell volume ratio
                            if overall_ratio < 0:
                                send_buy = False
                                msg = f"Not sending a buy order because overall ratio is {overall_ratio}"
                                if logger:
                                    logger.debug(msg)
                                else:
                                    print(msg)
                                client.cancel_all_asset(order['token'])
                            else:
                                # Place new buy order if any of these conditions are met:
                                # 1. We can get a better price than current order
                                if best_bid > orders['buy']['price']:
                                    msg = f"Sending Buy Order for {token} because better price. Orders look like this: {orders['buy']}. Best Bid: {best_bid}"
                                    if logger:
                                        logger.info(msg)
                                    else:
                                        print(msg)
                                    send_buy_order(order, logger)
                                # 2. Current position + orders is not enough to reach max_size
                                elif position + orders['buy']['size'] < 0.95 * max_size:
                                    msg = f"Sending Buy Order for {token} because not enough position + size"
                                    if logger:
                                        logger.info(msg)
                                    else:
                                        print(msg)
                                    send_buy_order(order, logger)
                                # 3. Our current order is too large and needs to be resized
                                elif orders['buy']['size'] > order['size'] * 1.01:
                                    msg = f"Resending buy orders because open orders are too large"
                                    if logger:
                                        logger.info(msg)
                                    else:
                                        print(msg)
                                    send_buy_order(order, logger)
                                # Commented out logic for cancelling orders when market conditions change
                                # elif best_bid_size < orders['buy']['size'] * 0.98 and abs(best_bid - second_best_bid) > 0.03:
                                #     print(f"Cancelling buy orders because best size is less than 90% of open orders and spread is too large")
                                #     global_state.client.cancel_all_asset(order['token'])
                        
                # ------- TAKE PROFIT / SELL ORDER MANAGEMENT -------
                elif sell_amount > 0:
                    # Sell the ENTIRE position, not just trade_size
                    # This handles cases where position > trade_size (oversized) or position < trade_size
                    order['size'] = position

                    # Skip if position is too small (dust)
                    if position < row['min_size'] * 0.5:
                        msg = f"Position {position} too small to sell (min_size: {row['min_size']}), skipping"
                        if logger:
                            logger.debug(msg)
                        else:
                            print(msg)
                        continue

                    # Calculate take-profit price based on average cost
                    tp_price = round_up(avgPrice + (avgPrice * params['take_profit_threshold']/100), round_length)

                    # Track market price history to detect trend
                    market_state_file = f'positions/{market}_{detail["name"]}_market.json'
                    previous_mid_price = None
                    if os.path.isfile(market_state_file):
                        try:
                            market_state = json.load(open(market_state_file))
                            previous_mid_price = market_state.get('mid_price')
                        except:
                            pass

                    # Save current mid_price for next iteration
                    json.dump({'mid_price': float(mid_price)}, open(market_state_file, 'w'))

                    # Determine if market is moving up, down, or stable
                    market_trend = 'unknown'
                    if previous_mid_price is not None:
                        price_change = mid_price - previous_mid_price
                        if price_change > 0.01:  # Moving up by more than 1 cent
                            market_trend = 'up'
                        elif price_change < -0.01:  # Moving down by more than 1 cent
                            market_trend = 'down'
                        else:
                            market_trend = 'stable'

                    # Calculate optimal sell price based on market conditions
                    # Start with a competitive price in the reward range
                    min_profitable_price = round_up(avgPrice * 1.01, round_length)  # At least 1% profit

                    # Competitive sell price: slightly above best ask to earn rewards
                    competitive_price = round_up(best_ask + row['tick_size'], round_length)

                    # If market is moving up, progressively move toward take-profit price
                    if market_trend == 'up':
                        # Interpolate between competitive price and tp_price based on how close we are
                        current_order_price = float(orders['sell']['price']) if orders['sell']['price'] > 0 else competitive_price
                        # Move 20% of the way toward tp_price each update when trending up
                        target_price = current_order_price + (tp_price - current_order_price) * 0.2
                        target_price = round_up(target_price, round_length)
                        msg = f"Market trending UP. Moving sell price from {current_order_price} toward tp_price {tp_price}. Target: {target_price}"
                        if logger:
                            logger.debug(msg)
                        else:
                            print(msg)
                    else:
                        # Market stable or down: use competitive pricing
                        target_price = max(competitive_price, min_profitable_price)
                        target_price = min(target_price, tp_price)  # Cap at take-profit
                        target_price = round_up(target_price, round_length)
                        msg = f"Market {market_trend}. Using competitive pricing. Target: {target_price} (best_ask: {best_ask}, min_profit: {min_profitable_price})"
                        if logger:
                            logger.debug(msg)
                        else:
                            print(msg)

                    order['price'] = target_price

                    tp_price = float(tp_price)
                    order_price = float(orders['sell']['price'])
                    target_price = float(target_price)

                    # Calculate % difference between current order and target price
                    diff = abs(order_price - target_price)/target_price * 100 if target_price > 0 else 100

                    # Update sell order if:
                    # 1. Current order price is significantly different from target (>2%)
                    if diff > 2:
                        msg = f"Sending Sell Order for {token} because current order price of {order_price} is deviant from target price of {target_price} and diff is {diff:.1f}%"
                        if logger:
                            logger.info(msg)
                        else:
                            print(msg)
                        send_sell_order(order, logger)
                    # 2. Current order size is too small for our position
                    elif orders['sell']['size'] < position * 0.97:
                        msg = f"Sending Sell Order for {token} because not enough sell size. Position: {position}, Sell Size: {orders['sell']['size']}"
                        if logger:
                            logger.info(msg)
                        else:
                            print(msg)
                        send_sell_order(order, logger)
                    # 3. Market is trending up and we should move price higher
                    elif market_trend == 'up' and order_price < target_price - row['tick_size']:
                        msg = f"Sending Sell Order for {token} to raise price as market trends up. Current: {order_price}, Target: {target_price}"
                        if logger:
                            logger.info(msg)
                        else:
                            print(msg)
                        send_sell_order(order, logger)
                    
                    # Commented out additional conditions for updating sell orders
                    # elif orders['sell']['price'] < ask_price:
                    #     print(f"Updating Sell Order for {token} because its not at the right price")
                    #     send_sell_order(order)
                    # elif best_ask_size < orders['sell']['size'] * 0.98 and abs(best_ask - second_best_ask) > 0.03...:
                    #     print(f"Cancelling sell orders because best size is less than 90% of open orders...")
                    #     send_sell_order(order)

        except Exception as ex:
            msg = f"Error performing trade for {market}: {ex}"
            if logger:
                logger.error(msg)
                logger.debug(traceback.format_exc())
            else:
                print(msg)
                traceback.print_exc()

        # Clean up memory and introduce a small delay
        gc.collect()
        await asyncio.sleep(2)