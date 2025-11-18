# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Poly-maker is a Polymarket automated trading system with multiple specialized bots. The main market-making bot uses WebSocket streaming for real-time trading, while utility bots handle specific strategies (near-sure, negative risk arbitrage, spike momentum).

## High-Level Architecture

### Core Components

**Main Market Maker (`main.py`)**
- Event-driven architecture using dual WebSocket streams:
  - Market data stream: price_change events update local order book (SortedDict)
  - User stream: tracks fills, cancellations, and position changes
- Global state management with thread-safe operations (`performing_trades` dict)
- Async market making with locks to prevent concurrent trades on same market
- Configuration via Google Sheets (fetched periodically)

**Trading Bots (in `near_sure/`, `neg_risk_arb/`, `spike_momentum/`)**
- Each bot has isolated credentials and operates independently
- `near_sure`: High-certainty opportunities (midpoint > 0.85 or < 0.15)
- `neg_risk_arb`: Arbitrage on negatively correlated binary events
- `spike_momentum`: Experimental post-resolution and news-driven strategies

**Position Merger (`merger/`)**
- Node.js script using Web3 to merge opposing positions on-chain
- Recovers capital locked in hedged positions (YES + NO on same market)
- Called periodically by main bot to optimize capital efficiency

**Shared Infrastructure (`poly_data/`, `poly_utils/`)**
- `PolymarketClient`: Wraps py-clob-client with order book fetching, balance checks
- `GoogleSheetsClient`: Real-time configuration database
- Logging system: structured JSON logs (trade/event/summary streams)
- Proxy support for remote server execution

### Key Design Patterns

**WebSocket Event Handling**
- `price_change` events update local order book (don't re-fetch via REST)
- Position tracking with race condition handling (user stream may lag trade execution)
- Heartbeat monitoring with automatic reconnection

**State Management**
- `performing_trades` dict prevents concurrent orders on same market
- `current_positions` tracks live positions with stop-loss/take-profit
- Order book caching in SortedDict for O(1) best bid/ask lookups

**Configuration as Code**
- Google Sheets stores market lists, spread targets, position limits
- Periodic refresh (60s) allows runtime config changes without restart
- Fallback to local defaults if Sheets unavailable

**Risk Management**
- Stop-loss: Exit if position moves against you beyond threshold
- Take-profit: Lock in gains at target profit level
- Volatility checks: Skip markets with wild price swings
- Position limits: Max positions per market and total exposure

## Development Commands

### Setup
```bash
# Install dependencies (uses UV package manager)
uv pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your credentials:
#   NEAR_SURE_PK, NEAR_SURE_BROWSER_ADDRESS (wallet private key + address)
#   PROXY settings if running on remote server
```

### Running Bots

**Main Market Maker**
```bash
python main.py
```

**Near-Sure Bot**
```bash
python near_sure/main.py trade          # Interactive mode
python near_sure/main.py monitor        # Continuous monitoring
python near_sure/main.py trade --dry-run  # Simulate trades
```

**Negative Risk Arbitrage**
```bash
python neg_risk_arb/main.py
```

**Spike Momentum Bot**
```bash
python spike_momentum/main.py
```

**Position Merger**
```bash
cd merger
node merger.js
```

### Testing & Debugging

**Logging**
- Logs written to `logs/` directory with rotation
- Three streams: trades (trade execution), events (market events), summary (daily P&L)
- See LOGGING.md for detailed logging configuration

**Dry Run Mode**
- Most bots support `--dry-run` flag to simulate trades without execution
- Useful for testing strategies against live market data

**Interactive Mode**
- Near-sure bot has interactive UI (InquirerPy) for manual market selection
- Fallback to simple stdin prompts if terminal incompatible

## Important Technical Details

### Position Tracking Race Conditions
The user WebSocket stream may deliver fill events AFTER the trade execution returns. Always check both:
1. Local state (`current_positions`)
2. REST API (`get_positions()`)

### Order Book Updates
- NEVER re-fetch order book on every `price_change` event
- Update local SortedDict cache incrementally
- Only full re-fetch on reconnection or staleness timeout

### Google Sheets as Database
- Sheet structure: each row = one market configuration
- Columns: market name, spread target, position limit, enabled flag
- Changes propagate within 60s (polling interval)

### Proxy Configuration
- Set `HTTPS_PROXY` in .env for remote server execution
- Required for WebSocket connections through HTTP proxies
- See PROXY_SETUP.md for detailed configuration

### Market Scanner Pre-Filtering
- Filter by closing time BEFORE fetching order books (performance optimization)
- Order book fetches are expensive; pre-filter reduces API calls by ~80%

### InquirerPy Compatibility
- Interactive prompts may fail on some terminals (tmux, ssh sessions)
- Always provide fallback to basic `input()` prompts
- Set `NEAR_SURE_SIMPLE_INPUT=1` to force simple mode

## Git Workflow

### Branch Merging
**ALWAYS use `--no-ff` when merging branches into main:**
```bash
git merge --no-ff <branch-name> -m "Merge branch '<branch-name>'"
```

This preserves branch structure in git history and makes it clear which commits were developed together as a feature.

### Commit Messages
Follow existing patterns: action verb + brief description
- "Fix TypeError: convert max_hours to float before comparison"
- "Add semantic search support for dramatically improved news matching"
- "Optimize market scanning with pre-filtering by closing time"
