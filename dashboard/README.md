# Poly-Maker Dashboard

This directory contains the local dashboard for managing the Poly-Maker bot.

## Architecture

The dashboard consists of two parts:
1.  **Backend (`server.py`)**: A FastAPI server that:
    *   Serves the frontend.
    *   Manages configuration (`config.json`).
    *   Fetches market data from Polymarket (replacing `update_markets.py`).
    *   Tracks account stats (replacing `update_stats.py`).
2.  **Frontend (`frontend/`)**: A React+Vite application for the UI.

## Setup & Usage

1.  **Install Dependencies**:
    ```bash
    uv sync
    cd dashboard/frontend && npm install && cd ../..
    ```

2.  **Start the Dashboard**:
    Use the helper script in the root directory:
    ```bash
    ./start_dashboard.sh
    ```
    This will start both the backend (port 8001) and frontend (port 5173).

3.  **Configuration**:
    *   The dashboard reads/writes to `config.json` in the root directory.
    *   Market data is cached in `dashboard/data/`.

## Key Files
*   `server.py`: Main backend entry point.
*   `data_updater/market_fetcher.py`: Logic for scanning/analyzing markets.
*   `poly_stats/account_stats.py`: Logic for calculating account PnL.
