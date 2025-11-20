import asyncio
import yaml
import logging
import sys
import os
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.getcwd())

from poly_data.polymarket_client import PolymarketClient
from poly_utils.logging_utils import get_logger
from sports_arb.market_scanner import SportsMarketScanner
from sports_arb.strategy import SportsArbStrategy

# Setup simple logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = get_logger('sports_arb.main')

async def main():
    print(f"\n{'='*60}")
    print("🏀 SPORTS ARB BOT - LIVE & POST-GAME")
    print(f"{'='*60}\n")
    
    # Load Config
    try:
        with open('sports_arb/config.yaml') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("config.yaml not found in sports_arb/")
        return

    strategy_cfg = config.get('strategy', {})
    logger.info(f"Mode: {'DRY RUN' if strategy_cfg.get('dry_run', True) else 'LIVE TRADING'}")

    # Init Client
    load_dotenv()
    try:
        client = PolymarketClient(account_type='sports_arb')
    except Exception as e:
        logger.error(f"Failed to init client: {e}")
        return

    # Init Components
    scanner = SportsMarketScanner(client, config)
    strategy = SportsArbStrategy(client, config)
    
    # 1. Initial Market Scan
    markets = await scanner.fetch_active_sports_markets()
    
    # 2. Start Background Score Polling
    asyncio.create_task(scanner.update_live_scores())
    
    # 3. Connect to WebSocket
    import websockets
    import json
    
    uri = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    
    # Build initial token map
    token_map = {}
    for m in scanner.markets.values():
        for t in m['tokens']:
            token_map[t['id']] = m
    
    while True:
        try:
            # Get prioritized token list
            tokens = scanner.get_websocket_subscriptions()
            if not tokens:
                logger.warning("No active tokens to monitor. Waiting...")
                await asyncio.sleep(60)
                continue
                
            logger.info(f"Connecting to WebSocket (Monitoring {len(tokens)} tokens)...")
            
            async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as ws:
                # Subscribe
                await ws.send(json.dumps({"assets_ids": tokens}))
                logger.info("✅ Subscribed!")
                
                # Refresh markets periodically
                last_refresh = asyncio.get_event_loop().time()
                
                while True:
                    # Check for market refresh
                    now = asyncio.get_event_loop().time()
                    if now - last_refresh > config['scanner']['refresh_interval']:
                        logger.info("Refreshing market list...")
                        await scanner.fetch_active_sports_markets()
                        
                        # Rebuild token map
                        token_map = {}
                        for m in scanner.markets.values():
                            for t in m['tokens']:
                                token_map[t['id']] = m
                                
                        # Reconnect to update subscriptions
                        break 
                    
                    msg = await ws.recv()
                    data = json.loads(msg)
                    
                    # Handle Updates
                    if isinstance(data, list):
                        for update in data:
                            await _handle_update(update, token_map, strategy)
                    else:
                        await _handle_update(data, token_map, strategy)
                        
        except Exception as e:
            logger.error(f"WebSocket Error: {e}")
            await asyncio.sleep(5)

async def _handle_update(update: dict, token_map: dict, strategy: SportsArbStrategy):
    if update.get('event_type') == 'book':
        asset_id = update.get('asset_id')
        bids = update.get('bids', [])
        asks = update.get('asks', [])
        
        if bids and asks:
            best_bid = float(bids[0]['price'])
            best_ask = float(asks[0]['price'])
            
            # O(1) Lookup
            found_market = token_map.get(asset_id)
            
            if found_market:
                await strategy.process_market_update(found_market, asset_id, best_bid, best_ask)

if __name__ == "__main__":
    asyncio.run(main())
