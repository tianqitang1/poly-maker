# Implementation Plan: Local Dashboard for Poly-Maker

## Goal Description
Replace the cumbersome Google Sheets interface with a modern, local web dashboard. This will allow the user to:
1.  **Browse Markets**: View available markets (fetched from Polymarket API or cached).
2.  **Select Markets**: Toggle markets for trading (replacing "Selected Markets" sheet).
3.  **Configure Parameters**: Edit hyperparameters for different strategies (replacing "Hyperparameters" sheet).
4.  **Persist Data**: Save configuration to a local `config.json` file, eliminating the need for `update_markets.py` and `update_stats.py` to sync with Google Sheets.

## User Review Required
> [!IMPORTANT]
> This change will decouple the bot from Google Sheets. The `SPREADSHEET_URL` environment variable will no longer be the source of truth.
> You will need to run the new dashboard server (`uv run python dashboard/server.py`) alongside your bot.

## Proposed Changes

### 1. Backend (FastAPI)
Create a new directory `dashboard/` with a FastAPI server.
*   **`dashboard/server.py`**:
    *   API to read/write `config.json`.
    *   API to fetch market data from Polymarket (proxying requests or using cached data).
    *   Serve the static frontend files.
*   **`dashboard/models.py`**: Pydantic models for configuration validation.

### 2. Frontend (React + Vite)
Create a `dashboard/frontend` directory.
*   **Market List**: Table view of markets with search/filter.
*   **Selection Toggle**: Easy switch to enable/disable trading for a market.
*   **Config Editor**: Form to edit hyperparameters (spread, size, etc.).

### 3. Bot Integration
Modify `poly_data/utils.py` to read from `config.json` instead of Google Sheets.
*   **`poly_data/utils.py`**:
    *   Update `get_sheet_df` to load from `config.json` if a specific flag/env var is set (e.g., `USE_LOCAL_CONFIG=true`).
    *   Maintain backward compatibility for now (optional).

### 4. Data Structure (`config.json`)
```json
{
  "selected_markets": [
    {
      "condition_id": "...",
      "question": "...",
      "token1": "...",
      "token2": "...",
      "param_type": "default",
      "min_size": 10,
      "max_size": 100,
      ...
    }
  ],
  "hyperparameters": {
    "default": {
      "spread": 0.01,
      ...
    }
  }
}
```

## Verification Plan

### Automated Tests
*   Unit tests for `config.json` reading/writing.
*   API tests for the dashboard endpoints.

### Manual Verification
1.  Start the dashboard server.
2.  Open `http://localhost:8000` in the browser.
3.  Select a market and save.
4.  Verify `config.json` is updated.
5.  Run `og_maker_kai` and verify it picks up the selected market from the local config.
