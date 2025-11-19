# Poly-Maker Context

## Project Overview
**Poly-Maker** is a comprehensive market making bot for **Polymarket** prediction markets. It is designed to provide liquidity by maintaining orders on both sides of the book, managing positions, and handling risk.

The project is modular, supporting multiple trading strategies and allowing for A/B testing. It integrates with **Google Sheets** for configuration (market selection, parameters) and uses **Websockets** for real-time interaction.

## Key Technologies
*   **Language**: Python 3.10+ (Primary), Node.js (for specific merging utilities).
*   **Package Manager**: `uv` (Python), `npm` (Node.js).
*   **Data/Config**: Google Sheets API, `.env` file.
*   **Architecture**:
    *   **Core Logic**: `poly_data/` (Data management, client interactions), `poly_utils/` (Shared utilities).
    *   **Strategies**:
        *   `og_maker/`: Original, simpler market making strategy.
        *   `too_clever_maker/`: Enhanced, more defensive strategy.
        *   `spike_momentum/`: Momentum-based strategy detecting price spikes.
        *   `neg_risk_arb/`: Negative risk arbitrage.
        *   `near_sure/`: High-probability betting strategy.
    *   **Utilities**: `poly_merger/` (Position merging via Node.js), `poly_stats/` (Statistics).

## Setup and Execution

### Dependencies
This project uses `uv` for Python dependency management.

1.  **Install Python dependencies**:
    ```bash
    uv sync
    ```
2.  **Install Node.js dependencies** (required for `poly_merger`):
    ```bash
    cd poly_merger && npm install && cd ..
    ```

### Configuration
*   **Environment Variables**: Create a `.env` file (based on `.env.example`) containing:
    *   Polymarket Private Keys (`PK`, `OG_MAKER_PK`, etc.)
    *   Wallet Addresses (`BROWSER_ADDRESS`, etc.)
    *   Google Sheets URL (`SPREADSHEET_URL`)
    *   Proxy settings (optional)
*   **Google Sheets**: The bot's behavior is controlled via a Google Sheet (tabs: "Selected Markets", "Hyperparameters", "All Markets").

### Running the Bot
*   **Main Market Maker**:
    ```bash
    uv run python main.py
    ```
*   **Specific Strategies** (for A/B testing):
    ```bash
    uv run python og_maker/main.py
    uv run python too_clever_maker/main.py
    ```
*   **Maintenance Scripts**:
    *   Update Market Data: `uv run python update_markets.py` (Should run continuously/periodically)
    *   Update Stats: `uv run python update_stats.py`

## Directory Structure
*   `main.py`: Entry point for the primary bot. Handles initialization, websockets, and the main update loop.
*   `poly_data/`: Handles Polymarket client interactions, websocket handlers, and global state (`global_state.py`).
*   `poly_utils/`: Utilities for logging, proxy config, and Google utils.
*   `poly_merger/`: JS-based tool for merging binary positions (YES + NO) to release collateral.
*   `logs/`: Stores application logs.
*   `STRATEGY_COMPARISON.md`: Documentation comparing different strategies implemented in the repo.

## Development Notes
*   **Concurrency**: Uses `asyncio` and `threading`. `main.py` runs a background thread for periodic updates (positions, orders) while the main thread handles websockets.
*   **Global State**: Heavily relies on `poly_data.global_state` to share data (positions, orders, market info) across modules.
*   **Proxies**: Configured in `poly_utils/proxy_config.py` and initialized early in `main.py`.
