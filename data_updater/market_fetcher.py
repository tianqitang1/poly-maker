import pandas as pd
from data_updater.find_markets import get_all_markets, get_all_results, get_markets, add_volatility_to_df
from data_updater.trading_utils import get_clob_client
from poly_data.polymarket_client import PolymarketClient
import os
import json

def get_local_selected_markets():
    if not os.path.exists('config.json'):
        return pd.DataFrame()
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        return pd.DataFrame(config.get('selected_markets', []))
    except:
        return pd.DataFrame()

def run_market_update(client=None):
    if client is None:
        client = get_clob_client()

    # 1. Get Selected Markets (from local config)
    sel_df = get_local_selected_markets()

    # 2. Get All Markets from API
    print("Fetching all markets...")
    all_df = get_all_markets(client)
    
    # 3. Process Results
    print("Processing market results...")
    all_results = get_all_results(all_df, client)
    
    # 4. Filter and Combine
    print("Combining markets...")
    m_data, all_markets = get_markets(all_results, sel_df, maker_reward=0.75)
    
    # 5. Add Volatility
    print("Adding volatility data...")
    new_df = add_volatility_to_df(all_markets)
    new_df['volatility_sum'] =  new_df['24_hour'] + new_df['7_day'] + new_df['14_day']
    
    new_df = new_df.sort_values('volatility_sum', ascending=True)
    new_df['volatilty/reward'] = ((new_df['gm_reward_per_100'] / new_df['volatility_sum']).round(2)).astype(str)

    # Select specific columns
    cols = ['question', 'answer1', 'answer2', 'spread', 'rewards_daily_rate', 'gm_reward_per_100', 'sm_reward_per_100', 'bid_reward_per_100', 'ask_reward_per_100',  'volatility_sum', 'volatilty/reward', 'min_size', '1_hour', '3_hour', '6_hour', '12_hour', '24_hour', '7_day', '30_day',  
                     'best_bid', 'best_ask', 'volatility_price', 'max_spread', 'tick_size',  
                     'neg_risk',  'market_slug', 'token1', 'token2', 'condition_id']
    
    # Filter only columns that exist
    existing_cols = [c for c in cols if c in new_df.columns]
    new_df = new_df[existing_cols]

    volatility_df = new_df.copy()
    volatility_df = volatility_df[new_df['volatility_sum'] < 20]
    volatility_df = volatility_df.sort_values('gm_reward_per_100', ascending=False)
    new_df = new_df.sort_values('gm_reward_per_100', ascending=False)

    # 6. Save to JSON
    os.makedirs("dashboard/data", exist_ok=True)
    
    def save_df(df, name):
        try:
            df.to_json(f"dashboard/data/{name}.json", orient="records")
        except Exception as e:
            print(f"Error saving {name}: {e}")

    save_df(new_df, "all_markets")
    save_df(volatility_df, "volatility_markets")
    save_df(m_data, "full_markets")
    
    print("Market update complete.")
    return new_df

if __name__ == "__main__":
    run_market_update()
