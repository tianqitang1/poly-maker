# Strategy Evaluation: OG Maker vs Too Clever Maker

I have analyzed the "very original bot" (commit `a6f87c5`) and compared it with your current `og_maker` and `too_clever_maker` implementations.

**Verification**: `og_maker` is indeed a faithful copy of the original strategy from commit `a6f87c5`.

Here is the evaluation of room for improvement for **both** strategies.

## 1. OG Maker (Original Strategy)

**Current State**: Simple, robust, partial stop-loss.
**Verdict**: Good foundation, but artificially limited.

### Room for Improvement

#### A. Unlock Two-Sided Market Making (High Impact)
The original code explicitly prevents holding positions on both YES and NO sides:
```python
# og_maker/trading.py
if rev_pos['size'] > row['min_size']:
    print("Bypassing creation of new buy order because there is a reverse position")
```
**Improvement**: **Remove this check.**
By allowing the bot to hold both sides, you can capture the spread on the entire market (delta neutral). If you buy YES at 0.40 and NO at 0.58 (equivalent to selling YES at 0.42), you lock in a profit regardless of the outcome. The current logic forces you to be directional.

#### B. Dynamic Take-Profit (Medium Impact)
Currently, it uses a static percentage:
```python
tp_price = avgPrice + (avgPrice * params['take_profit_threshold']/100)
```
**Improvement**: **Implement Volatility-Adjusted Take-Profit.**
If the market is volatile (`row['3_hour']` is high), increase the threshold to capture more upside. If volatility is low, decrease it to ensure you get out with a profit before the price stagnates.

---

## 2. Too Clever Maker (Enhanced Strategy)

**Current State**: Defensive, "paranoid", full stop-loss, trend tracking.
**Verdict**: Over-engineered in ways that hurt performance.

### Room for Improvement

#### A. Fix "Chasing" Trend Logic (High Impact)
The current logic is counter-productive for a market maker:
```python
# too_clever_maker/trading.py
if market_trend == 'up':
    # Raises sell price, moving AWAY from the liquidity
    target_price = current_order_price + (tp_price - current_order_price) * 0.2
```
**Critique**: When price spikes up, liquidity takers want to buy. You should be **selling into** this demand, not raising your price and running away.
**Improvement**: **Invert or Remove.** If the market trends up, keep your sell price competitive (at `best_ask`) to guarantee a fill. You make money on the spread/turnover, not by holding for a home run.

#### B. Soften the Stop-Loss (High Impact)
Currently, it sells 100% of the position on a dip:
```python
pos_to_sell = position  # Entire position
```
**Critique**: This locks in losses on temporary dips (whipsaws).
**Improvement**: **Use Partial Stop-Loss (like OG Maker).**
Sell only 50% (or `trade_size`) initially. This reduces exposure while leaving room for the price to recover (mean reversion).

#### C. Simplify "Dead Zones" (Medium Impact)
The bot has too many conflicting thresholds (`0.9`, `0.95`, `1.1`, `0.8` for reverse).
**Improvement**: **Consolidate logic.**
Use a single `TargetPosition` and a simple tolerance band. If `current < Target - band`, buy. If `current > Target + band`, sell.

## Summary of Proposed Changes

| Strategy | Change | Benefit |
| :--- | :--- | :--- |
| **Both** | **Remove Reverse Position Check** | Allows profitable two-sided market making (capturing spread). |
| **OG Maker** | **Dynamic Take-Profit** | Adapts to market volatility for better exits. |
| **Too Clever** | **Fix Trend Tracking** | Prevents "chasing" price; ensures fills during spikes. |
| **Too Clever** | **Partial Stop-Loss** | Prevents full exit on temporary dips; reduces realized losses. |

**Recommendation**: I suggest applying these changes to `too_clever_maker` first to transform it into a "Smart Maker" that combines the robustness of OG with better logic, while keeping `og_maker` as a stable baseline.
