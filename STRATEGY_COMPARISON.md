# Strategy Comparison: OG Maker vs Too Clever Maker

## Overview

Two new bots created for A/B testing different market making strategies:
- **og_maker**: Original simpler strategy (commit a6f87c5)
- **too_clever_maker**: Current enhanced defensive strategy

## Key Strategy Differences

### Position Sizing

| Aspect | OG Maker | Too Clever Maker |
|--------|----------|------------------|
| Buy condition | `position < max_size` | `position < max_size * 0.95` |
| Pre-emptive cancel | None | Cancels buys at 90% max_size |
| Threshold flexibility | Simple | Multiple safety margins |

### Stop-Loss Behavior

| Aspect | OG Maker | Too Clever Maker |
|--------|----------|------------------|
| Amount sold | `trade_size` (partial) | Entire position |
| Recovery ability | Can rebuild from remaining | Must start fresh |
| Capital efficiency | Keeps some exposure | Exits completely |

**Impact**: OG allows gradual unwinding, Too Clever forces full exit on drawdown

### Entry Logic

| Aspect | OG Maker | Too Clever Maker |
|--------|----------|------------------|
| Reverse position check | After main logic | Early check (lines 615-634) |
| Threshold | `> min_size` | `>= 80% max_size OR > max(min_size, 10% max_size)` |
| Effect | Allows balanced making | Prevents opposing positions |

**Impact**: OG can hold both YES and NO positions profitably, Too Clever avoids this

### Take-Profit Strategy

| Aspect | OG Maker | Too Clever Maker |
|--------|----------|------------------|
| Pricing | Fixed `avgPrice + threshold%` | Dynamic market trend tracking |
| Complexity | Simple calculation | Tracks price history, interpolates |
| Adaptability | Static | Adjusts to market momentum |

**Impact**: OG has predictable exits, Too Clever tries to capture more upside but may miss fills

### Over-Sizing Prevention

| Aspect | OG Maker | Too Clever Maker |
|--------|----------|------------------|
| Layers of checks | 1 (position < max_size) | 4+ (pre-cancel, position check, order validation, balance checks) |
| Safety | Basic | Paranoid |

**Impact**: Too Clever has "dead zones" where it won't trade near limits

## Why Too Clever Might Feel "Clunky and Goes Against You"

### 1. **Too Defensive on Entries**
```python
# Too Clever (lines 615-634)
if rev_pos['size'] >= max_size * 0.8 or rev_pos['size'] > max(row['min_size'], max_size * 0.1):
    # Don't buy - opposing position exists
```
- Prevents profitable two-sided market making
- Misses opportunities when both outcomes have positive expected value

### 2. **Market Trend Tracking Issues**
```python
# Too Clever (lines 774-794)
if market_trend == 'up':
    # Move sell price higher toward take-profit
    target_price = current_order_price + (tp_price - current_order_price) * 0.2
```
- Raises sell price when market moves up
- May hold too long if market reverses
- Misses fills waiting for higher prices

### 3. **Over-Sizing Paranoia**
- Multiple 0.95, 0.9, 0.8 thresholds create complexity
- "Dead zones" where bot is uncertain
- May exit profitable positions too early

### 4. **Full Position Stop-Loss**
```python
# Too Clever (line 543)
pos_to_sell = position  # Entire position

# OG Maker
pos_to_sell = sell_amount  # Just trade_size
```
- Too Clever exits completely on temporary dips
- Can't benefit from mean reversion
- Higher transaction costs (full exit + re-entry vs partial)

## Setup Instructions

### 1. Add Credentials to `.env`

```bash
# OG Maker
OG_MAKER_PK=your_private_key_here
OG_MAKER_BROWSER_ADDRESS=your_wallet_address_here

# Too Clever Maker
TOO_CLEVER_MAKER_PK=your_private_key_here
TOO_CLEVER_MAKER_BROWSER_ADDRESS=your_wallet_address_here
```

**Recommendation**: Use separate wallets for clean A/B testing

### 2. Run Bots

```bash
# Terminal 1: OG Maker
python og_maker/main.py

# Terminal 2: Too Clever Maker
python too_clever_maker/main.py
```

### 3. Monitor Performance

Track these metrics for each bot:
- Total P&L
- Win rate
- Average profit per trade
- Position turnover (entries/exits per day)
- Capital utilization
- Stop-loss frequency
- Fill rate (orders placed vs filled)

## Expected Outcomes

### When OG Maker Should Win
- Markets with balanced opportunities on both sides (YES and NO both profitable)
- Stable markets with mean reversion
- When transaction costs are significant (fewer exits = lower costs)
- Markets where holding through small drawdowns is profitable

### When Too Clever Maker Should Win
- One-sided trending markets
- High volatility environments
- Markets with frequent sharp reversals (stop-loss saves capital)
- When precise position limits are critical

## Hypothesis to Test

**Primary Hypothesis**: The original simpler strategy (OG Maker) will outperform the enhanced defensive strategy (Too Clever Maker) because:

1. Market making profits from providing liquidity on both sides
2. Temporary drawdowns often mean-revert (partial stop-loss > full exit)
3. Simpler logic = fewer edge cases = more consistent behavior
4. Less defensive checks = more trades = more opportunities to earn spread

**Counter-Hypothesis**: Too Clever Maker will win if markets are trending/volatile, where defensive logic prevents large losses.

## Next Steps

1. **Run both bots for 1-2 weeks** on same markets with equal capital
2. **Track detailed metrics** (create logs/comparison_metrics.csv)
3. **Analyze edge cases** where each strategy excels or fails
4. **Iterate**: Combine best elements of both strategies
5. **Consider LLM-aided approach** that can switch between aggressive/defensive based on market conditions

## Files

- `og_maker/main.py` - OG bot entry point
- `og_maker/trading.py` - Original strategy logic
- `og_maker/README.md` - OG strategy documentation
- `too_clever_maker/main.py` - Too Clever bot entry point
- `too_clever_maker/trading.py` - Enhanced strategy logic
- `too_clever_maker/README.md` - Too Clever strategy documentation
- `.env.example` - Updated with new bot credentials
- `STRATEGY_COMPARISON.md` - This file

## Future: LLM-Aided Approach

After gathering data from both strategies, consider building `llm_maker` that:
- Uses LLM to select which strategy mode (aggressive vs defensive)
- Adjusts position limits dynamically based on market conditions
- Provides directional bias for asymmetric market making
- Adapts stop-loss behavior based on market regime

See `STRATEGY_COMPARISON.md` for initial LLM maker design.
