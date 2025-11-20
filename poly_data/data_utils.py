import poly_data.global_state as global_state
from poly_data.utils import get_sheet_df
import time
import poly_data.global_state as global_state

#sth here seems to be removing the position
def update_positions(avgOnly=False):
    pos_df = global_state.client.get_all_positions()

    # Track which tokens are in the current API response
    api_tokens = set()

    for idx, row in pos_df.iterrows():
        asset = str(row['asset'])
        api_tokens.add(asset)

        if asset in  global_state.positions:
            position = global_state.positions[asset].copy()
        else:
            position = {'size': 0, 'avgPrice': 0}

        position['avgPrice'] = row['avgPrice']

        if not avgOnly:
            position['size'] = row['size']
        else:
            # Only block position updates for a SHORT time after last trade update
            # This prevents indefinite blocking when trades get stuck in 'performing'

            try:
                old_size = position['size']
            except:
                old_size = 0

            # Check if we have any pending trades
            has_pending_trades = False
            for col in [f"{asset}_sell", f"{asset}_buy"]:
                if col in global_state.performing and isinstance(global_state.performing[col], set) and len(global_state.performing[col]) > 0:
                    has_pending_trades = True
                    break

            # Determine if we should update position
            should_update = False

            if not has_pending_trades:
                # No pending trades - always safe to update
                should_update = True
            elif asset in global_state.last_trade_update:
                # Has pending trades - only block updates for 10 seconds after last trade
                time_since_last_trade = time.time() - global_state.last_trade_update[asset]

                if time_since_last_trade < 10:
                    # Recent trade - block update to avoid race conditions
                    print(f"Skipping update for {asset} - recent trade {time_since_last_trade:.1f}s ago, pending trades exist")
                    should_update = False
                else:
                    # Trade is old (>10s) - force update even with pending trades
                    # This handles stuck trades that never get CONFIRMED
                    print(f"FORCING position update for {asset} - {time_since_last_trade:.1f}s since last trade, but trades still pending (likely stuck)")
                    should_update = True
            else:
                # Has pending trades but no last_trade_update timestamp - update anyway
                print(f"Updating position for {asset} - pending trades exist but no timestamp recorded")
                should_update = True

            if should_update and old_size != row['size']:
                print(f"Updating position from {old_size} to {row['size']} and avgPrice to {row['avgPrice']} using API")
                position['size'] = row['size']

        global_state.positions[asset] = position

    # Clear positions for tokens that are no longer in the API response
    # This handles cases where positions were fully exited (sold completely)
    if not avgOnly:
        tokens_to_clear = []
        for token in list(global_state.positions.keys()):
            if token not in api_tokens and global_state.positions[token]['size'] > 0:
                # Check if there are pending trades for this token
                has_pending_trades = False
                for col in [f"{token}_sell", f"{token}_buy"]:
                    if col in global_state.performing and isinstance(global_state.performing[col], set) and len(global_state.performing[col]) > 0:
                        has_pending_trades = True
                        break

                # Only clear if no pending trades (position was truly exited)
                if not has_pending_trades:
                    tokens_to_clear.append(token)

        # Clear the positions
        for token in tokens_to_clear:
            old_size = global_state.positions[token]['size']
            old_avg = global_state.positions[token]['avgPrice']
            global_state.positions[token] = {'size': 0, 'avgPrice': 0}
            print(f"✓ Cleared stale position for token {token}: was {old_size} @ {old_avg}, now 0 (not in API response)")

def get_position(token):
    token = str(token)
    if token in global_state.positions:
        return global_state.positions[token]
    else:
        return {'size': 0, 'avgPrice': 0}

def set_position(token, side, size, price, source='websocket'):
    token = str(token)
    size = float(size)
    price = float(price)

    global_state.last_trade_update[token] = time.time()
    
    if side.lower() == 'sell':
        size *= -1

    if token in global_state.positions:
        
        prev_price = global_state.positions[token]['avgPrice']
        prev_size = global_state.positions[token]['size']


        if size > 0:
            if prev_size == 0:
                # Starting a new position
                avgPrice_new = price
            else:
                # Buying more; update average price
                avgPrice_new = (prev_price * prev_size + price * size) / (prev_size + size)
        elif size < 0:
            # Selling; average price remains the same
            avgPrice_new = prev_price
        else:
            # No change in position
            avgPrice_new = prev_price


        global_state.positions[token]['size'] += size
        global_state.positions[token]['avgPrice'] = avgPrice_new
    else:
        global_state.positions[token] = {'size': size, 'avgPrice': price}

    print(f"Updated position from {source}, set to ", global_state.positions[token])

def update_orders():
    all_orders = global_state.client.get_all_orders()
    print(f"DEBUG: API returned {len(all_orders)} total orders")

    # Create new orders dict starting with what we find in the API
    api_orders = {}

    if len(all_orders) > 0:
        # Ensure asset_id is string for consistent filtering
        all_orders['asset_id'] = all_orders['asset_id'].astype(str)

        for token in all_orders['asset_id'].unique():
            # Fix: Ignore tokens that are not in our selected markets
            if str(token) not in global_state.all_tokens:
                continue
            
            token_str = str(token)
            if token_str not in api_orders:
                api_orders[token_str] = {'buy': {'price': 0, 'size': 0}, 'sell': {'price': 0, 'size': 0}}

            curr_orders = all_orders[all_orders['asset_id'] == token_str]
            
            if len(curr_orders) > 0:
                sel_orders = {}
                sel_orders['buy'] = curr_orders[curr_orders['side'] == 'BUY']
                sel_orders['sell'] = curr_orders[curr_orders['side'] == 'SELL']

                for type in ['buy', 'sell']:
                    curr = sel_orders[type]

                    if len(curr) > 1:
                        print(f"DEBUG: Found {len(curr)} {type} orders for {token}. Cancelling all.")
                        print(f"Multiple {type} orders found for {token}, cancelling")
                        global_state.client.cancel_all_asset(token)
                        api_orders[token_str][type] = {'price': 0, 'size': 0}
                    elif len(curr) == 1:
                        api_orders[token_str][type]['price'] = float(curr.iloc[0]['price'])
                        api_orders[token_str][type]['size'] = float(curr.iloc[0]['original_size'] - curr.iloc[0]['size_matched'])

    # Now merge with global_state.orders to preserve recently updated orders (grace period)
    current_time = time.time()
    merged_orders = api_orders.copy()

    # Check for orders we know about locally but API missed
    for token, sides in global_state.orders.items():
        token = str(token)
        if token not in merged_orders:
            merged_orders[token] = {'buy': {'price': 0, 'size': 0}, 'sell': {'price': 0, 'size': 0}}
        
        for side in ['buy', 'sell']:
            local_side = sides.get(side, {})
            # If we have a local order with a recent timestamp
            if local_side.get('size', 0) > 0 and 'last_update' in local_side:
                # If updated within last 10 seconds
                if current_time - local_side['last_update'] < 10:
                    # If API shows no order, but we have a recent one, KEEP LOCAL
                    if merged_orders[token][side]['size'] == 0:
                        merged_orders[token][side] = local_side
                        # print(f"Preserving recent local {side} order for {token} (API missed it)")
    
    global_state.orders = merged_orders

def get_order(token):
    token = str(token)
    if token in global_state.orders:

        if 'buy' not in global_state.orders[token]:
            global_state.orders[token]['buy'] = {'price': 0, 'size': 0}

        if 'sell' not in global_state.orders[token]:
            global_state.orders[token]['sell'] = {'price': 0, 'size': 0}

        return global_state.orders[token]
    else:
        return {'buy': {'price': 0, 'size': 0}, 'sell': {'price': 0, 'size': 0}}
    
def set_order(token, side, size, price):
    token = str(token)
    side = side.lower()

    # Ensure the token exists in the orders dict, creating it if it doesn't
    if token not in global_state.orders:
        global_state.orders[token] = {'buy': {'price': 0, 'size': 0}, 'sell': {'price': 0, 'size': 0}}
    
    # Update the specific side of the order with timestamp
    global_state.orders[token][side]['size'] = float(size)
    global_state.orders[token][side]['price'] = float(price)
    global_state.orders[token][side]['last_update'] = time.time()

    print(f"Updated order for token {token}, state is now: ", global_state.orders[token])

    

def update_markets():
    received_df, received_params = get_sheet_df()

    if len(received_df) > 0:
        global_state.df, global_state.params = received_df.copy(), received_params
        global_state.MARKET_TOKENS = {}

    for _, row in global_state.df.iterrows():
        for col in ['token1', 'token2']:
            row[col] = str(row[col])

        # Track token1/token2 for this market to orient websocket books
        cid = str(row['condition_id'])
        global_state.MARKET_TOKENS[cid] = {'token1': row['token1'], 'token2': row['token2']}

        if row['token1'] not in global_state.all_tokens:
            global_state.all_tokens.append(row['token1'])
        
        if row['token2'] not in global_state.all_tokens:
            global_state.all_tokens.append(row['token2'])

        if row['token1'] not in global_state.REVERSE_TOKENS:
            global_state.REVERSE_TOKENS[row['token1']] = row['token2']

        if row['token2'] not in global_state.REVERSE_TOKENS:
            global_state.REVERSE_TOKENS[row['token2']] = row['token1']

        for col2 in [f"{row['token1']}_buy", f"{row['token1']}_sell", f"{row['token2']}_buy", f"{row['token2']}_sell"]:
            if col2 not in global_state.performing:
                global_state.performing[col2] = set()
