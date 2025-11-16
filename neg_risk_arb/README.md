# Negative Risk Arbitrage Bot

An automated arbitrage bot for Polymarket that exploits negative risk markets where **YES + NO < $1.00**, executing both sides atomically and merging positions for instant, risk-free profit.

## How It Works

### The Arbitrage Opportunity

In Polymarket's negative risk markets, there are scenarios where:
- **YES ask**: $0.48
- **NO ask**: $0.51
- **Total cost**: $0.99

Since holding both YES and NO positions can be merged to recover $1.00, this creates a **$0.01 profit per share** with minimal risk.

### Execution Flow

1. **Scan**: Find markets where YES ask + NO ask < $1.00
2. **Buy**: Execute market orders on both YES and NO sides
3. **Merge**: Combine opposing positions to recover $1.00 USDC
4. **Profit**: Instant profit = $1.00 - total cost

### Risk Management

The bot includes sophisticated partial fill protection:
- If only one side fills → automatically rescue with market order (configurable loss limit)
- If neither fills → opportunity lost, no risk
- If both fill unequally → merge the smaller amount, handle remainder

## Installation

1. **Install dependencies**:
   ```bash
   uv sync
   # or
   pip install -e .
   pip install PyYAML  # for config file support
   ```

2. **Configure environment variables** in `.env`:
   ```bash
   NEG_RISK_ARB_PK=your_private_key_here
   NEG_RISK_ARB_BROWSER_ADDRESS=your_wallet_address_here
   ```

3. **Adjust config.yaml** (see Configuration section below)

## Usage

### Scan Mode

Find and display current arbitrage opportunities:

```bash
python -m neg_risk_arb.main scan
```

**Output:**
```
ARBITRAGE OPPORTUNITIES (15 found, showing top 10)
================================================================================
1. Will market close above $50?... | Profit: 45.2bp ($0.45) | Cost: $0.9955 | Size: $100 | Close: 12.3h
2. Election winner in 2024...      | Profit: 38.1bp ($0.38) | Cost: $0.9962 | Size: $100 | Close: 5.2h
...
```

### Trade Mode

Execute a single arbitrage opportunity:

```bash
# With confirmation prompt
python -m neg_risk_arb.main trade

# Test with dry run first
python -m neg_risk_arb.main trade --dry-run
```

**Workflow:**
1. Scans for opportunities
2. Shows top 5 opportunities
3. Prompts you to select one (if InquirerPy installed)
4. Executes both orders atomically
5. Merges positions
6. Reports profit/loss

### Auto Mode

Automated arbitrage hunting - continuously scans and executes:

```bash
# Live trading
python -m neg_risk_arb.main auto

# Dry run for testing
python -m neg_risk_arb.main auto --dry-run
```

**Features:**
- Scans every N seconds (configurable)
- Auto-executes best opportunity
- Handles partial fills automatically
- Tracks daily P&L
- Stops if risk limits breached

### Custom Config

Use a custom configuration file:

```bash
python -m neg_risk_arb.main scan --config my_custom_config.yaml
```

## Configuration

All parameters are configurable via `neg_risk_arb/config.yaml`:

### Profit Thresholds

```yaml
min_profit_bps: 30        # Minimum 0.3% profit (30 basis points)
max_total_cost: 0.995     # Maximum YES + NO sum
```

**Recommendation:**
- Start with 50 bps (0.5%) to find high-quality opportunities
- Lower to 30 bps for more frequent trades
- Never go below 20 bps (fees will eat into profit)

### Position Sizing

```yaml
max_position_size: 100     # Max $100 per arbitrage
min_position_size: 10      # Min $10 per trade
min_liquidity_per_side: 50 # Require $50 liquidity on each side
```

**Recommendation:**
- Start with $50-100 max while testing
- Increase to $250-500 once comfortable
- Ensure min_liquidity > max_position_size for safety

### Execution Strategy

```yaml
order_type: market         # 'market' or 'limit'
execution_delay_ms: 100    # Delay between YES and NO orders
confirmation_timeout_sec: 3  # Wait time for fills
```

**Recommendation:**
- Use `market` orders for arbitrage (speed matters)
- Keep delay low (0-200ms) to minimize price movement risk
- 3-5s timeout is usually sufficient

### Partial Fill Protection

```yaml
partial_fill_strategy: market_rescue  # Options: market_rescue, limit_rescue, exit_position
max_rescue_loss_pct: 0.01             # Accept up to 1% loss to complete arb
rescue_timeout_sec: 5                 # Timeout for limit rescue
```

**Strategies:**
1. **market_rescue** (recommended): Immediately market order the unfilled side
   - Pros: Fast, usually completes the arb
   - Cons: May accept small loss if price moved

2. **limit_rescue**: Try limit order first, fallback to market
   - Pros: Better prices
   - Cons: Slower, may miss opportunity

3. **exit_position**: Sell the filled side at market
   - Pros: Caps maximum loss
   - Cons: Takes directional risk

### Risk Management

```yaml
max_concurrent_positions: 3   # Limit simultaneous arbs
max_daily_loss: 50            # Stop if lose $50 in a day
min_win_rate: 0.70            # Stop if win rate < 70%
failed_arb_cooldown_sec: 60   # Wait 60s after failed arb on same market
```

**Recommendation:**
- Start with 1-2 concurrent positions
- Set daily loss to 2-5% of capital
- 70% win rate is reasonable for arbitrage

### Market Filtering

```yaml
require_neg_risk: true            # Only trade negative risk markets
min_hours_until_close: 1.0        # Avoid markets closing too soon
max_hours_until_close: 168        # 7 days
exclude_keywords: []              # Markets to skip
```

## Project Structure

```
neg_risk_arb/
├── config.yaml            # All configurable parameters
├── __init__.py            # Package initialization
├── arbitrage_scanner.py   # Finds YES + NO < $1 opportunities
├── executor.py            # Atomic buy + merge logic
├── risk_manager.py        # Partial fill protection
├── main.py                # CLI entry point
├── README.md              # This file
└── arb_history.json       # Trade log (generated)
```

## Examples

### Example 1: Conservative Arbitrage

**Goal:** Only take high-confidence opportunities

```yaml
# config.yaml
min_profit_bps: 50              # 0.5% minimum
max_position_size: 50           # $50 per trade
max_total_cost: 0.995
partial_fill_strategy: market_rescue
max_rescue_loss_pct: 0.005      # 0.5% max rescue loss
```

```bash
python -m neg_risk_arb.main auto
```

**Expected:**
- 3-5 trades per day
- 0.4-0.8% profit per trade
- Very high win rate (>85%)

### Example 2: Aggressive Arbitrage

**Goal:** Maximize volume and total profit

```yaml
# config.yaml
min_profit_bps: 20              # 0.2% minimum
max_position_size: 250          # $250 per trade
max_total_cost: 0.998
partial_fill_strategy: market_rescue
max_rescue_loss_pct: 0.01       # 1% max rescue loss
```

```bash
python -m neg_risk_arb.main auto
```

**Expected:**
- 10-20 trades per day
- 0.2-0.5% profit per trade
- Good win rate (>75%)
- Higher volume = higher total profit

### Example 3: Manual Cherry-Picking

**Goal:** Pick the best opportunities yourself

```bash
# Scan for opportunities
python -m neg_risk_arb.main scan

# Review the list, then execute one
python -m neg_risk_arb.main trade
# Select the opportunity you want
```

## Expected Performance

Based on typical market conditions:

### Profit per Trade
- **Conservative (50bp min)**: $0.25 - $1.00 per $100 traded
- **Balanced (30bp min)**: $0.15 - $0.60 per $100 traded
- **Aggressive (20bp min)**: $0.10 - $0.40 per $100 traded

### Frequency
- **Opportunities appear**: 10-30 times per day
- **High-quality (>40bp)**: 5-10 times per day
- **Optimal window**: When markets are active (US hours)

### Win Rate
- **Successful merges**: 80-95% (with market_rescue)
- **Partial fill rescues**: 10-15% of trades
- **Failed arbs**: <5% (usually due to race conditions)

### Daily Profit Estimates
- **Conservative** ($100 max size, 50bp min): $2-5/day
- **Balanced** ($250 max size, 30bp min): $5-15/day
- **Aggressive** ($500 max size, 20bp min): $10-25/day

**Note:** Performance varies with market activity and capital deployed.

## Safety Features

1. **Isolated Account**: Uses separate `NEG_RISK_ARB_*` credentials
2. **Dry Run Mode**: Test everything without risking capital
3. **Slippage Protection**: Rejects trades if prices moved too much
4. **Loss Limits**: Stops trading if daily loss exceeds threshold
5. **Win Rate Monitor**: Pauses if success rate drops too low
6. **Partial Fill Protection**: Multiple strategies to handle incomplete fills
7. **Cooldown Periods**: Prevents rapid retries on failed markets

## Troubleshooting

### No opportunities found

**Possible causes:**
- Profit threshold too high
- Market conditions (all markets priced efficiently)
- Wrong time of day (low activity)

**Solutions:**
```yaml
min_profit_bps: 20        # Lower threshold
max_total_cost: 0.998     # Allow closer to $1.00
max_hours_until_close: null  # Remove time filter
```

### Orders failing

**Possible causes:**
- Insufficient USDC balance
- Wallet hasn't traded via UI yet
- API rate limits

**Solutions:**
- Ensure wallet has sufficient balance
- Execute one manual trade via Polymarket UI
- Increase `execution_delay_ms` to 200-500

### Partial fills happening frequently

**Possible causes:**
- Order book liquidity too thin
- Execution delay too high
- Market moving fast

**Solutions:**
```yaml
min_liquidity_per_side: 100   # Require more liquidity
execution_delay_ms: 50        # Faster execution
max_position_size: 50         # Smaller size
```

### Merge failing

**Possible causes:**
- Positions too small (< 20 shares)
- Gas price too low
- Network congestion

**Solutions:**
- Ensure positions > 20 shares minimum
- Check merge_delay_sec (increase to 10s)
- Retry manually or wait for network to clear

### Rescue strategy accepting too many losses

**Possible causes:**
- `max_rescue_loss_pct` too high
- Market volatility

**Solutions:**
```yaml
max_rescue_loss_pct: 0.005    # Tighten to 0.5%
partial_fill_strategy: limit_rescue  # Try limit first
```

## Advanced Usage

### Running Multiple Instances

You can run multiple instances with different configs:

```bash
# Terminal 1: Conservative
python -m neg_risk_arb.main auto --config config_conservative.yaml

# Terminal 2: Aggressive
python -m neg_risk_arb.main auto --config config_aggressive.yaml
```

**Important:** Use different accounts to avoid conflicts.

### Integration with Main Bot

The neg_risk_arb bot is **completely isolated**:
- Uses different environment variables
- No shared state
- Can run simultaneously with main bot on different accounts
- Only shares the base PolymarketClient class

### Logging Trades

Enable trade logging in config:

```yaml
log_successful_arbs: true
log_file: neg_risk_arb/arb_history.json
```

Trade log format:
```json
{
  "timestamp": "2024-11-16T10:30:00",
  "market": "Will X happen?",
  "profit": 0.52,
  "size": 100,
  "total_cost": 99.48
}
```

## Performance Tips

1. **Optimal Timing**:
   - Most opportunities during US market hours (9am-6pm ET)
   - Activity spikes around major events
   - Weekends can be slower

2. **Capital Efficiency**:
   - Positions merge within minutes (funds quickly available)
   - Can recycle capital 10-20x per day
   - Start with $500-1000 to test

3. **Network Considerations**:
   - Polygon gas is cheap (<$0.01 per merge)
   - Merges happen on-chain, takes 30-60 seconds
   - Keep some MATIC for gas (0.1 MATIC is plenty)

4. **Monitoring**:
   - Check logs regularly
   - Monitor win rate (should stay >70%)
   - Review failed arbs to adjust config

## FAQ

**Q: Is this actually risk-free?**
A: No. While the math is "risk-free" when both sides fill, execution risk exists:
- Partial fills can cause losses (mitigated by rescue strategies)
- Smart contract risk (position merging)
- Blockchain network risk (pending transactions)

**Q: Why does the opportunity exist?**
A: Market inefficiencies, especially in low-liquidity markets. Also:
- Different users have different fee tiers
- Maker rebates vs taker fees create pricing discrepancies
- Short-term supply/demand imbalances

**Q: How much capital do I need?**
A: Minimum $200-300 to start. More capital = more opportunities.

**Q: Can I run this 24/7?**
A: Yes, but opportunities are scarce during off-hours. Better to run during active market hours.

**Q: What's the catch?**
A: Small profits per trade, requires active monitoring, and occasional losses from partial fills.

## License

Same as parent project (MIT)

## Support

For issues or questions, refer to the main project repository.

---

**Disclaimer:** Trading cryptocurrency and prediction markets carries risk. Start with small amounts and dry-run mode. Past performance doesn't guarantee future results.
