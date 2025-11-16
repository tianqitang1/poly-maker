# Near-Sure Trading Utility

A specialized trading utility for Polymarket that focuses on near-certain market outcomes with passive order placement and comprehensive risk management.

## Features

- 🔍 **Smart Market Scanning**: Automatically scans all Polymarket markets for near-certain outcomes (prices close to 0 or 1)
- 🎯 **Interactive Selection**: Beautiful terminal UI with arrow key navigation for market selection
- 📊 **Passive Order Placement**: Places limit orders at existing bid prices near the midpoint (no aggressive market orders)
- 🛡️ **Account-Wide Stop Loss**: Monitors all positions and implements automatic stop-loss protection
- 💼 **Dedicated Account**: Isolated from main bot using separate environment variables

## Installation

1. **Install dependencies** (from project root):
   ```bash
   pip install -e .
   ```

   Or manually:
   ```bash
   pip install InquirerPy==0.3.4
   ```

2. **Configure environment variables** in your `.env` file:
   ```bash
   # Near-sure account credentials (separate from main bot)
   NEAR_SURE_PK=your_private_key_here
   NEAR_SURE_BROWSER_ADDRESS=your_wallet_address_here
   ```

## Usage

### Interactive Trading Mode

Scan markets, select interactively, and place orders:

```bash
python -m near_sure.main trade
```

**Interactive workflow:**
1. Configure filter parameters (price thresholds, time until close, etc.)
2. Browse filtered markets with arrow keys
3. Select markets with spacebar
4. Enter trade amounts for each market
5. Review and confirm

### Risk Monitoring Mode

Continuously monitor all positions with stop-loss protection:

```bash
python -m near_sure.main monitor
```

**Options:**
```bash
# Custom stop loss (-15%)
python -m near_sure.main monitor --stop-loss -0.15

# Check every 60 seconds
python -m near_sure.main monitor --check-interval 60
```

### Combined Mode

Place trades then automatically start monitoring:

```bash
python -m near_sure.main trade-monitor
```

### Dry Run Mode

Test without placing real orders:

```bash
python -m near_sure.main trade --dry-run
python -m near_sure.main monitor --dry-run
```

## How It Works

### Market Filtering

Markets are filtered based on:

- **Price Certainty**: Midpoint price ≥ 0.85 (bullish) or ≤ 0.15 (bearish)
- **Certainty Score**: Calculated as `1 - min(price, 1-price) / 0.5`
- **Closing Time**: Markets closing within specified timeframe
- **Minimum Hours**: Avoid markets closing too soon (default: 1 hour)

### Passive Order Strategy

Instead of crossing the spread with market orders, the utility:

1. Fetches the full order book
2. Calculates the midpoint
3. Finds the closest existing bid price ≤ midpoint
4. Places a limit order at that price
5. Ensures minimum order size compliance

**Example:**
```
Order Book:
  Best Bid: 0.89 (size: 50)
  Midpoint: 0.895
  Best Ask: 0.90 (size: 100)

Strategy:
  ✓ Place BUY order at 0.89 (passive, at existing bid)
  ✗ NOT at 0.90 (would cross spread aggressively)
```

### Stop-Loss Protection

The risk manager:

1. Monitors all positions every N seconds (default: 30s)
2. Calculates unrealized PnL for each position
3. Triggers stop loss if PnL% ≤ threshold (default: -10%)
4. Executes market sell order to exit position
5. Cancels pending orders for stopped positions

## Project Structure

```
near_sure/
├── __init__.py           # Package initialization
├── market_scanner.py     # Market discovery and filtering
├── interactive_ui.py     # Terminal-based UI with InquirerPy
├── order_manager.py      # Passive order placement logic
├── risk_manager.py       # Stop-loss monitoring
├── main.py               # CLI entry point
└── README.md             # This file
```

## Command Reference

### Main Commands

```bash
# Show help
python -m near_sure.main --help

# Interactive trading
python -m near_sure.main trade [--dry-run]

# Risk monitoring
python -m near_sure.main monitor [--check-interval N] [--stop-loss PCT] [--dry-run]

# Combined mode
python -m near_sure.main trade-monitor [--check-interval N] [--stop-loss PCT] [--dry-run]
```

### Arguments

- `--dry-run`: Simulate without placing orders or executing stop losses
- `--check-interval N`: Seconds between position checks (default: 30)
- `--stop-loss PCT`: Stop loss percentage, e.g., -0.10 for -10% (default: -0.10)

## Configuration Options

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `NEAR_SURE_PK` | Private key for near-sure account | Yes |
| `NEAR_SURE_BROWSER_ADDRESS` | Wallet address for near-sure account | Yes |

### Filter Parameters

When using interactive mode, you can configure:

- **Min Midpoint**: Minimum price for bullish near-sure (default: 0.85)
- **Max Midpoint**: Maximum price for bearish near-sure (default: 0.15)
- **Max Hours Until Close**: Maximum hours before market closes (default: 48)
- **Min Hours Until Close**: Minimum hours to avoid imminent closes (default: 1)
- **Min Certainty Score**: Minimum certainty threshold 0-1 (default: 0.70)

## Examples

### Example 1: Conservative Near-Sure Trading

```bash
# Trade with dry run first to test
python -m near_sure.main trade --dry-run

# If satisfied, run live
python -m near_sure.main trade

# Interactive prompts will guide you through:
# 1. Set min_midpoint = 0.90 (very confident outcomes)
# 2. Set max_hours = 24 (closing within 24 hours)
# 3. Select markets with spacebar
# 4. Enter $100 for each market
# 5. Confirm and place orders
```

### Example 2: Aggressive Monitoring

```bash
# Monitor with tighter stop loss and frequent checks
python -m near_sure.main monitor --stop-loss -0.05 --check-interval 15
```

### Example 3: Full Workflow

```bash
# Place trades and immediately start monitoring
python -m near_sure.main trade-monitor
```

## Safety Features

1. **Isolated Account**: Uses separate credentials (`NEAR_SURE_*`) to avoid conflicts with main bot
2. **Dry Run Mode**: Test all functionality without risking capital
3. **Position Validation**: Checks for existing positions before trading
4. **Minimum Size Enforcement**: Respects market minimum order sizes
5. **Order Book Validation**: Ensures valid prices before placing orders
6. **Stop Loss Protection**: Automatic exit on adverse price movements

## Troubleshooting

### "InquirerPy not installed"

```bash
pip install InquirerPy==0.3.4
```

### "Missing environment variables"

Ensure your `.env` file contains:
```bash
NEAR_SURE_PK=your_private_key
NEAR_SURE_BROWSER_ADDRESS=your_wallet_address
```

### No markets found

Try adjusting filter parameters:
- Lower `min_midpoint` (e.g., 0.80 instead of 0.85)
- Increase `max_hours_until_close` (e.g., 72 instead of 48)
- Lower `min_certainty` threshold (e.g., 0.60 instead of 0.70)

### Orders failing

- Ensure wallet has sufficient USDC balance
- Check that wallet has executed at least one trade via Polymarket UI (for API access)
- Verify the wallet address matches the private key
- Check for minimum order size violations

## Development

### Running Tests

```bash
# Test with dry run
python -m near_sure.main trade --dry-run
python -m near_sure.main monitor --dry-run
```

### Adding New Features

The modular structure makes it easy to extend:

- **Market Filtering**: Edit `market_scanner.py` → `filter_near_sure_markets()`
- **Order Logic**: Edit `order_manager.py` → `find_passive_bid_price()`
- **Risk Rules**: Edit `risk_manager.py` → `check_stop_loss_trigger()`
- **UI Options**: Edit `interactive_ui.py` → `get_filter_parameters()`

## Integration with Main Bot

The near-sure utility is **completely isolated** from the main bot:

- Uses different environment variables (`NEAR_SURE_*`)
- No shared state or global variables
- Can run simultaneously with main bot on different accounts
- Does not modify main bot code (except adding `use_near_sure` parameter)

## Performance Tips

1. **Market Scanning**: First scan can take 2-3 minutes for all markets
2. **Order Placement**: Add delay between orders (default: 1s) to avoid rate limits
3. **Monitoring**: 30s interval is recommended; faster may hit API limits
4. **Position Count**: Works best with 5-15 concurrent positions

## License

Same as parent project (MIT)

## Support

For issues or questions, please refer to the main project repository.
