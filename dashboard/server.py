from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import json
import os
import sys
import pandas as pd
import asyncio
import time
import threading

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from poly_data.polymarket_client import PolymarketClient
from poly_utils.proxy_config import setup_proxy
from data_updater.market_fetcher import run_market_update
from poly_stats.account_stats import update_stats_once
from dotenv import load_dotenv

load_dotenv()
setup_proxy()

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
CONFIG_FILE = "config.json"

# Initialize config if not exists
if not os.path.exists(CONFIG_FILE):
    default_config = {
        "selected_markets": [],
        "hyperparameters": {
            "default": {
                "spread": 0.01,
                "min_size": 10,
                "max_size": 100,
                "trade_size": 50
            }
        }
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(default_config, f, indent=4)

# --- Background Tasks ---
client = None # Global client

def background_update_task():
    """
    Periodically updates markets and stats.
    """
    global client
    if client is None:
        try:
            client = PolymarketClient()
        except Exception as e:
            print(f"Error init client in bg task: {e}")
            return

    while True:
        print("Running background updates...")
        try:
            # Update Stats
            update_stats_once(client)
            
            # Update Markets (Every 1 hour roughly, but here we can do it less often or on demand)
            # Since market update is heavy, maybe just do stats frequently.
            # Let's do stats every 5 mins.
        except Exception as e:
            print(f"Error in background stats update: {e}")
        
        time.sleep(300) # 5 minutes

# Thread for background updates
bg_thread = threading.Thread(target=background_update_task, daemon=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global client
    try:
        client = PolymarketClient()
    except Exception as e:
        print(f"Warning: Could not initialize PolymarketClient: {e}")
    
    bg_thread.start()
    
    yield
    # Shutdown (nothing specific needed)

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_json_data(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return []

# --- Data Endpoints ---

@app.get("/data/stats")
def get_stats():
    return get_json_data("stats.json")

@app.get("/data/all_markets")
def get_all_markets_data():
    return get_json_data("all_markets.json")

@app.get("/data/volatility_markets")
def get_volatility_markets_data():
    return get_json_data("volatility_markets.json")

@app.post("/actions/refresh_markets")
def trigger_market_refresh(background_tasks: BackgroundTasks):
    """
    Trigger a heavy market update (takes time).
    """
    def task():
        print("Starting manual market refresh...")
        try:
            run_market_update(client)
        except Exception as e:
            print(f"Error in manual market refresh: {e}")
    
    background_tasks.add_task(task)
    return {"status": "started", "message": "Market refresh started in background"}

@app.post("/actions/refresh_stats")
def trigger_stats_refresh(background_tasks: BackgroundTasks):
    def task():
        print("Starting manual stats refresh...")
        try:
            update_stats_once(client)
        except Exception as e:
            print(f"Error in manual stats refresh: {e}")
            
    background_tasks.add_task(task)
    return {"status": "started", "message": "Stats refresh started in background"}

# --- Config Endpoints (Local JSON) ---

@app.get("/config/sheets/selected_markets")
def get_selected_markets():
    # We keep the endpoint name /sheets/ for frontend compatibility for now, 
    # or we should update frontend. Let's keep it simple and just change behavior.
    # Actually, the user wanted "Local Dashboard", so let's be explicit.
    # The frontend calls this.
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
        return config.get("selected_markets", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/config/sheets/selected_markets")
def save_selected_markets(markets: list[dict]):
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
        
        config["selected_markets"] = markets
        
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)
            
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config/sheets/hyperparameters")
def get_hyperparams():
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
        
        # Transform to list of dicts if frontend expects that (it seemed to expect list of dicts for sheets)
        # But wait, the sheets format was row-based (type, param, value).
        # My config.json structure is nested dict: {"type": {"param": value}}.
        # I should convert to sheets format for frontend compatibility OR update frontend.
        # Let's update frontend to handle the cleaner JSON structure.
        # BUT, to minimize frontend changes right now, I will return the cleaner structure
        # and update the frontend to match.
        return config.get("hyperparameters", {})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/config/sheets/hyperparameters")
def save_hyperparams(params: dict): # Changed to dict
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
        
        config["hyperparameters"] = params
        
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)
            
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/markets")
def get_markets(limit: int = 50, cursor: str = ""):
    """
    Fetch active markets from Polymarket API directly (for browsing).
    """
    if not client:
        raise HTTPException(status_code=503, detail="Polymarket client not initialized")
    
    try:
        response = client.client.get_sampling_markets(next_cursor=cursor)
        
        markets = []
        for m in response.get('data', []):
            tokens = m.get('tokens', [])
            if len(tokens) < 2:
                continue
                
            market_entry = {
                "condition_id": m.get('condition_id'),
                "question": m.get('question'),
                "token1": tokens[0].get('token_id'),
                "token2": tokens[1].get('token_id'),
                "slug": m.get('market_slug'),
                "end_date": m.get('end_date_iso'),
            }
            markets.append(market_entry)
            
        return {
            "markets": markets,
            "next_cursor": response.get('next_cursor')
        }
        
    except Exception as e:
        print(f"Error fetching markets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
