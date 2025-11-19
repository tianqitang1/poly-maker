import json
from poly_utils.google_utils import get_spreadsheet
import pandas as pd 
import os

def pretty_print(txt, dic):
    print("\n", txt, json.dumps(dic, indent=4))

def get_sheet_df(read_only=None):
    """
    Get sheet data with optional read-only mode
    
    Args:
        read_only (bool): If None, auto-detects based on credentials availability
    """
    all = 'All Markets'
    sel = 'Selected Markets'

    # Check for local config first
    if os.path.exists('config.json'):
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
            
            # Convert selected_markets list to DataFrame
            df = pd.DataFrame(config.get('selected_markets', []))
            
            # Get hyperparameters
            hyperparams = config.get('hyperparameters', {})
            
            print("Loaded configuration from local config.json")

            # Attempt to merge with cached market data if available
            # This mimics the merge with "All Markets" sheet
            all_markets_path = os.path.join(os.path.dirname(__file__), '..', 'dashboard', 'data', 'all_markets.json')
            if os.path.exists(all_markets_path):
                try:
                    with open(all_markets_path, 'r') as f:
                        all_markets_data = json.load(f)
                    
                    if all_markets_data:
                        df2 = pd.DataFrame(all_markets_data)
                        # Merge on question
                        # Note: config.json might already have some fields, but we trust all_markets_json for latest stats
                        # We use suffix to avoid collisions if needed, but usually we want to enrich
                        
                        # Only merge if we have questions to match
                        if not df.empty and 'question' in df.columns:
                            # Drop columns in df that might duplicate what's in df2 (except the join key)
                            # Actually, we usually trust config for 'params' and df2 for 'market specs'
                            # Let's just do a standard merge
                            result = df.merge(df2, on='question', how='inner', suffixes=('', '_y'))
                            
                            # Remove duplicate columns if any (preferring the left one usually, but here right has stats)
                            # Actually, let's trust the merge.
                            print(f"Merged with local market data. Result: {len(result)} markets.")
                            return result, hyperparams
                        else:
                             print("Config has no selected markets or missing 'question' column.")
                             return df, hyperparams

                except Exception as ex:
                    print(f"Error reading/merging all_markets.json: {ex}")
            else:
                print("Warning: dashboard/data/all_markets.json not found. Market stats (rewards, volatility) may be missing.")

            return df, hyperparams
        except Exception as e:
            print(f"Error reading config.json: {e}. Falling back to Google Sheets.")

    # Auto-detect read-only mode if not specified
    if read_only is None:
        creds_file = 'credentials.json' if os.path.exists('credentials.json') else '../credentials.json'
        read_only = not os.path.exists(creds_file)
        if read_only:
            print("No credentials found, using read-only mode")

    try:
        spreadsheet = get_spreadsheet(read_only=read_only)
    except FileNotFoundError:
        print("No credentials found, falling back to read-only mode")
        spreadsheet = get_spreadsheet(read_only=True)

    wk = spreadsheet.worksheet(sel)
    df = pd.DataFrame(wk.get_all_records())
    df = df[df['question'] != ""].reset_index(drop=True)

    wk2 = spreadsheet.worksheet(all)
    df2 = pd.DataFrame(wk2.get_all_records())
    df2 = df2[df2['question'] != ""].reset_index(drop=True)

    result = df.merge(df2, on='question', how='inner')

    wk_p = spreadsheet.worksheet('Hyperparameters')
    records = wk_p.get_all_records()
    hyperparams, current_type = {}, None

    for r in records:
        # Update current_type only when we have a non-empty type value
        # Handle both string and NaN values from pandas
        type_value = r['type']
        if type_value and str(type_value).strip() and str(type_value) != 'nan':
            current_type = str(type_value).strip()
        
        # Skip rows where we don't have a current_type set
        if current_type:
            # Convert numeric values to appropriate types
            value = r['value']
            try:
                # Try to convert to float if it's numeric
                if isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                    value = float(value)
                elif isinstance(value, (int, float)):
                    value = float(value)
            except (ValueError, TypeError):
                pass  # Keep as string if conversion fails
            
            hyperparams.setdefault(current_type, {})[r['param']] = value

    return result, hyperparams
