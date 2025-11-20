import math 
from poly_data.data_utils import update_positions
import poly_data.global_state as global_state

# def get_avgPrice(position, assetId):
#     curr_global = global_state.all_positions[global_state.all_positions['asset'] == str(assetId)]
#     api_position_size = 0
#     api_avgPrice = 0

#     if len(curr_global) > 0:
#         c_row = curr_global.iloc[0]
#         api_avgPrice = round(c_row['avgPrice'], 2)
#         api_position_size = c_row['size']

#     if position > 0:
#         if abs((api_position_size - position)/position * 100) > 5:
#             print("Updating global positions")
#             update_positions()

#             try:
#                 c_row = curr_global.iloc[0]
#                 api_avgPrice = round(c_row['avgPrice'], 2)
#                 api_position_size = c_row['size']
#             except:
#                 return 0
#     return api_avgPrice

def get_best_bid_ask_deets(market, name, size, deviation_threshold=0.05):
    market_data = global_state.all_data[market]
    # Assume the incoming book is always for token1 (YES). Invert only when reading token2.
    invert_prices = (name == 'token2')

    best_bid, best_bid_size, second_best_bid, second_best_bid_size, top_bid = find_best_price_with_size(market_data['bids'], size, reverse=True)
    best_ask, best_ask_size, second_best_ask, second_best_ask_size, top_ask = find_best_price_with_size(market_data['asks'], size, reverse=False)
    
    # Handle None values in mid_price calculation
    if best_bid is not None and best_ask is not None:
        mid_price = (best_bid + best_ask) / 2
        bid_sum_within_n_percent = sum(size for price, size in market_data['bids'].items() if best_bid <= price <= mid_price * (1 + deviation_threshold))
        ask_sum_within_n_percent = sum(size for price, size in market_data['asks'].items() if mid_price * (1 - deviation_threshold) <= price <= best_ask)
    else:
        mid_price = None
        bid_sum_within_n_percent = 0
        ask_sum_within_n_percent = 0

    # If the book looks inverted (bid > ask), flip once to correct orientation
    if best_bid is not None and best_ask is not None and best_bid > best_ask:
        invert_prices = not invert_prices
        print(f"[book flip] market {market} name {name}: bid {best_bid} > ask {best_ask}, flipping orientation")

    if invert_prices:
        # Handle None values before arithmetic operations
        if all(x is not None for x in [best_bid, best_ask, second_best_bid, second_best_ask, top_bid, top_ask]):
            best_bid, second_best_bid, top_bid, best_ask, second_best_ask, top_ask = 1 - best_ask, 1 - second_best_ask, 1 - top_ask, 1 - best_bid, 1 - second_best_bid, 1 - top_bid
            best_bid_size, second_best_bid_size, best_ask_size, second_best_ask_size = best_ask_size, second_best_ask_size, best_bid_size, second_best_bid_size
            bid_sum_within_n_percent, ask_sum_within_n_percent = ask_sum_within_n_percent, bid_sum_within_n_percent
        else:
            # Handle case where some prices are None - use available values or defaults
            if best_bid is not None and best_ask is not None:
                best_bid, best_ask = 1 - best_ask, 1 - best_bid
                best_bid_size, best_ask_size = best_ask_size, best_bid_size
            if second_best_bid is not None:
                second_best_bid = 1 - second_best_bid
            if second_best_ask is not None:
                second_best_ask = 1 - second_best_ask
            if top_bid is not None:
                top_bid = 1 - top_bid
            if top_ask is not None:
                top_ask = 1 - top_ask
            bid_sum_within_n_percent, ask_sum_within_n_percent = ask_sum_within_n_percent, bid_sum_within_n_percent



    #return as dictionary
    return {
        'best_bid': best_bid,
        'best_bid_size': best_bid_size,
        'second_best_bid': second_best_bid,
        'second_best_bid_size': second_best_bid_size,
        'top_bid': top_bid,
        'best_ask': best_ask,
        'best_ask_size': best_ask_size,
        'second_best_ask': second_best_ask,
        'second_best_ask_size': second_best_ask_size,
        'top_ask': top_ask,
        'bid_sum_within_n_percent': bid_sum_within_n_percent,
        'ask_sum_within_n_percent': ask_sum_within_n_percent
    }


def find_best_price_with_size(price_dict, min_size, reverse=False):
    lst = list(price_dict.items())

    if reverse:
        lst.reverse()
    
    best_price, best_size = None, None
    second_best_price, second_best_size = None, None
    top_price = None
    set_best = False

    for price, size in lst:
        if top_price is None:
            top_price = price

        if set_best:
            second_best_price, second_best_size = price, size
            break

        if size >= min_size:
            if best_price is None:
                best_price, best_size = price, size
                set_best = True

    return best_price, best_size, second_best_price, second_best_size, top_price

def _get_offset_ticks(row):
    """
    Safely pull the optional quote_offset_ticks column from the Selected Markets sheet.
    Falls back to 0 when the column is missing or empty.
    """
    offset = 0

    if hasattr(row, 'get'):
        offset = row.get('quote_offset_ticks', 0)
    elif isinstance(row, dict):
        offset = row.get('quote_offset_ticks', 0)

    try:
        offset = float(offset)
        if math.isnan(offset):
            return 0
    except (TypeError, ValueError):
        return 0

    return max(0, offset)

def get_order_prices(best_bid, best_bid_size, top_bid,  best_ask, best_ask_size, top_ask, avgPrice, row):

    tick_size = row['tick_size']
    bid_price = best_bid + tick_size
    ask_price = best_ask - tick_size

    if best_bid_size < row['min_size'] * 1.5:
        bid_price = best_bid
    
    if best_ask_size < 250 * 1.5:
        ask_price = best_ask
    
    offset_ticks = _get_offset_ticks(row)
    offset_amount = offset_ticks * tick_size

    if offset_amount > 0:
        if bid_price is not None:
            bid_price -= offset_amount
        if ask_price is not None:
            ask_price += offset_amount

    if bid_price >= top_ask:
        bid_price = top_bid

    if ask_price <= top_bid:
        ask_price = top_ask

    if bid_price == ask_price:
        bid_price = top_bid
        ask_price = top_ask

    mid_price = None
    if top_bid is not None and top_ask is not None:
        mid_price = (top_bid + top_ask) / 2

    if mid_price is not None and 'max_spread' in row:
        incentive_band = row['max_spread'] / 100
        floor_price = mid_price - incentive_band
        ceiling_price = mid_price + incentive_band

        if bid_price < floor_price:
            bid_price = floor_price
        if ask_price > ceiling_price:
            ask_price = ceiling_price

    if bid_price < 0:
        bid_price = 0
    if ask_price > 1:
        ask_price = 1

    #temp for sleep
    if ask_price <= avgPrice and avgPrice > 0:
        ask_price = avgPrice

    return bid_price, ask_price




def round_down(number, decimals):
    factor = 10 ** decimals
    return math.floor(number * factor) / factor

def round_up(number, decimals):
    factor = 10 ** decimals
    return math.ceil(number * factor) / factor

def get_buy_sell_amount(position, bid_price, row, other_token_position=0):
    buy_amount = 0
    sell_amount = 0

    # Get max_size, defaulting to trade_size if not specified
    max_size = row.get('max_size', row['trade_size'])
    trade_size = row['trade_size']
    
    # Calculate total exposure across both sides
    total_exposure = position + other_token_position
    
    # If we haven't reached max_size on either side, continue building
    # Use 0.95 threshold to account for floating point precision and partial fills
    if position < max_size * 0.95:
        # Continue quoting trade_size amounts until we reach max_size
        remaining_to_max = max_size - position
        buy_amount = min(trade_size, remaining_to_max)

        # Only sell if we have substantial position (to allow for exit when needed)
        if position >= trade_size * 0.5:
            sell_amount = min(position, trade_size)
        else:
            sell_amount = 0
    else:
        # We've reached max_size, implement progressive exit strategy
        # Always offer to sell trade_size amount when at max_size
        sell_amount = min(position, trade_size)

        # Only continue buying if we're truly market making (have position on other side too)
        # This prevents over-sizing on one side while allowing balanced market making
        if other_token_position >= trade_size and total_exposure < max_size * 1.5:
            # Both sides have positions, allow flexibility for market making
            buy_amount = trade_size
        else:
            # Maxed out on one side only, stop buying
            buy_amount = 0

    # Ensure minimum order size compliance
    if buy_amount > 0.7 * row['min_size'] and buy_amount < row['min_size']:
        buy_amount = row['min_size']

    # Apply multiplier for low-priced assets (optional)
    if bid_price < 0.1 and buy_amount > 0:
        multiplier = row.get('multiplier', '')
        if multiplier != '' and multiplier is not None:
            print(f"Multiplying buy amount by {int(multiplier)}")
            buy_amount = buy_amount * int(multiplier)

    return buy_amount, sell_amount
