# OG Maker Kai Bot

An enhanced version of the original market making strategy (`og_maker`), incorporating specific improvements to maximize profitability and capital efficiency.

## Key Improvements

### 1. True Two-Sided Market Making (Delta Neutral)
- **Removed Reverse Position Check**: The original bot prevented buying if you held a position on the opposite side. `og_maker_kai` removes this restriction, allowing you to hold both YES and NO positions simultaneously.
- **Benefit**: Captures the spread on the entire market. If you buy YES at 0.40 and NO at 0.58, you lock in a profit regardless of the outcome.

### 2. Volatility-Adjusted Take-Profit
- **Dynamic Thresholds**: Instead of a static take-profit percentage, `og_maker_kai` adjusts the target based on market volatility (`3_hour` metric).
- **Benefit**:
    - **High Volatility**: Increases the take-profit target to capture more upside during big moves.
    - **Low Volatility**: Keeps the standard target to ensure consistent turnover.

## Setup

Uses the same credentials as `og_maker`:
```bash
OG_MAKER_PK=your_private_key_here
OG_MAKER_BROWSER_ADDRESS=your_wallet_address_here
```

## Running

```bash
uv run python og_maker_kai/main.py
```

## Comparison

| Feature | OG Maker (Original) | OG Maker Kai (Enhanced) |
|---------|---------------------|-------------------------|
| **Reverse Positions** | Blocked (One-sided only) | **Allowed (Two-sided / Delta Neutral)** |
| **Take-Profit** | Fixed % | **Dynamic (Volatility Adjusted)** |
| **Stop-Loss** | Partial (trade_size) | Partial (trade_size) |
| **Position Sizing** | Simple < max_size | Simple < max_size |
