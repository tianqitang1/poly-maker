# Logging System Documentation

The poly-maker project includes a comprehensive logging system that provides both detailed debugging logs and trade-specific action logs.

## Features

### 📊 **Dual Logging Streams**

1. **Detailed Log** (`*_detailed.log`)
   - Everything the bot does (DEBUG level)
   - Includes all print statements, calculations, decisions
   - Rotates at 10MB, keeps 5 backups
   - Great for debugging issues

2. **Trade Log** (`*_trades.log`)
   - Only significant events (orders, fills, merges, stop losses)
   - Rotates daily, keeps 30 days
   - Perfect for reviewing trading activity

3. **Events Log** (`*_events.jsonl`)
   - Structured JSON Lines format
   - Easy to parse and analyze
   - Each line is a complete JSON event

4. **Daily Summary** (`*_daily_summary.jsonl`)
   - Daily P&L and statistics
   - Event breakdowns
   - Performance metrics

### 📁 **Log Directory Structure**

```
logs/
├── main/                          # Main market-making bot
│   ├── main_detailed.log
│   ├── main_trades.log
│   ├── main_events.jsonl
│   └── main_daily_summary.jsonl
├── near_sure/                     # Near-sure trading bot
│   ├── near_sure_detailed.log
│   ├── near_sure_trades.log
│   ├── near_sure_events.jsonl
│   └── near_sure_daily_summary.jsonl
└── neg_risk_arb/                  # Negative risk arbitrage bot
    ├── neg_risk_arb_detailed.log
    ├── neg_risk_arb_trades.log
    ├── neg_risk_arb_events.jsonl
    └── neg_risk_arb_daily_summary.jsonl
```

## Usage

### Basic Setup

```python
from poly_utils.logging_utils import get_logger

# Initialize logger for your bot
logger = get_logger('my_bot_name')

# Use standard logging methods
logger.debug("Detailed debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error message")
```

### Trade-Specific Logging

```python
# Log an order placement
logger.log_order(
    action='BUY',
    market='Will X happen?',
    token='1234567890',
    price=0.65,
    size=100.0,
    order_id='abc123'  # Additional fields
)

# Log an order fill
logger.log_fill(
    action='BUY',
    market='Will X happen?',
    token='1234567890',
    fill_price=0.65,
    fill_size=100.0
)

# Log order cancellation
logger.log_cancel(
    market='Will X happen?',
    token='1234567890',
    reason='Price moved'
)

# Log position merge
logger.log_merge(
    market='Will X happen?',
    size=100.0,
    recovered=100.0
)

# Log stop loss
logger.log_stop_loss(
    market='Will X happen?',
    position=100.0,
    entry_price=0.65,
    exit_price=0.58,
    pnl=-7.0,
    reason='PnL below threshold'
)

# Log arbitrage execution
logger.log_arbitrage(
    market='Will X happen?',
    yes_price=0.487,
    no_price=0.508,
    size=100.0,
    profit=0.50,
    success=True
)
```

### Daily Summaries

```python
# Generate summary (returns dict)
summary = logger.generate_daily_summary()

# Save summary to file
logger.save_daily_summary()

# Print summary to console
logger.print_summary()
```

## Integration Examples

### Example 1: Main Bot Integration

```python
# In main.py
from poly_utils.logging_utils import get_logger

# Initialize logger at startup
logger = get_logger('main')

# Replace print statements with logger calls
# Before:
print(f"Creating order for {size} at {price}")

# After:
logger.info(f"Creating order for {size} at {price}")

# Log trades
logger.log_order(
    action='BUY',
    market=row['question'],
    token=token,
    price=bid_price,
    size=buy_amount
)
```

### Example 2: Integrating into Existing Bot

```python
# trading.py - Add at the top
from poly_utils.logging_utils import get_logger

# In perform_trade function
def perform_trade(market, row, logger=None):
    if logger is None:
        logger = get_logger('main')

    # Replace prints with logging
    logger.debug(f"Processing market: {row['question']}")

    # Log orders
    if buy_amount > 0:
        response = client.create_order(...)
        logger.log_order(
            action='BUY',
            market=row['question'],
            token=token,
            price=bid_price,
            size=buy_amount
        )

    # Log stop losses
    if pnl < stop_loss_threshold:
        logger.log_stop_loss(
            market=row['question'],
            position=position,
            entry_price=avgPrice,
            exit_price=exit_price,
            pnl=pnl_dollars,
            reason=f"PnL: {pnl:.2f}%"
        )
```

### Example 3: Near-Sure Bot Integration

```python
# near_sure/main.py
from poly_utils.logging_utils import get_logger

def trade_mode(dry_run=False):
    # Initialize logger
    logger = get_logger('near_sure')

    # ... existing code ...

    # Log order placement
    response = client.create_order(...)
    logger.log_order(
        action='BUY',
        market=opportunity['question'],
        token=token_id,
        price=price,
        size=size
    )

    # At end of session
    logger.save_daily_summary()
    logger.print_summary()
```

## Log Analysis

### Parsing JSON Events

```python
import json

# Read events log
with open('logs/neg_risk_arb/neg_risk_arb_events.jsonl', 'r') as f:
    events = [json.loads(line) for line in f]

# Filter arbitrage events
arbs = [e for e in events if e['event_type'] == 'arbitrage']

# Calculate total profit
total_profit = sum(e['profit'] for e in arbs if e['success'])
print(f"Total arbitrage profit: ${total_profit:.2f}")

# Win rate
success_count = sum(1 for e in arbs if e['success'])
win_rate = success_count / len(arbs) if arbs else 0
print(f"Win rate: {win_rate*100:.1f}%")
```

### Daily Summary Analysis

```python
import json
import pandas as pd

# Read all summaries
with open('logs/neg_risk_arb/neg_risk_arb_daily_summary.jsonl', 'r') as f:
    summaries = [json.loads(line) for line in f]

# Convert to DataFrame
df = pd.DataFrame(summaries)

# Analyze
print(f"Total P&L: ${df['total_pnl'].sum():.2f}")
print(f"Best day: ${df['total_pnl'].max():.2f}")
print(f"Worst day: ${df['total_pnl'].min():.2f}")
print(f"Avg daily P&L: ${df['total_pnl'].mean():.2f}")
```

### Searching Logs

```bash
# Find all errors
grep "ERROR" logs/main/main_detailed.log

# Find all arbitrage successes
grep "ARBITRAGE SUCCESS" logs/neg_risk_arb/neg_risk_arb_trades.log

# Count stop losses
grep "STOP LOSS" logs/main/main_trades.log | wc -l

# Extract all merges from JSON
jq 'select(.event_type == "merge")' logs/*/neg_risk_arb_events.jsonl
```

## Configuration

### Custom Log Levels

```python
# More verbose console output
logger = get_logger('my_bot', console_level='DEBUG')

# Less verbose file output
logger = get_logger('my_bot', file_level='INFO')

# Custom log directory
logger = get_logger('my_bot', log_dir='custom_logs')
```

### Log Rotation Settings

The rotation settings are configured in `logging_utils.py`:

- **Detailed log**: 10MB per file, 5 backups (50MB total)
- **Trade log**: Daily rotation, 30 days retention
- **Events/Summary**: No automatic rotation (append only)

## Best Practices

### 1. **Use Appropriate Log Levels**

```python
logger.debug("Calculating bid price...")           # Detailed calculations
logger.info("Placing order")                        # Significant actions
logger.warning("Price moved significantly")         # Potential issues
logger.error("Order placement failed")              # Errors
logger.critical("Cannot connect to API")            # Critical failures
```

### 2. **Log Trade Events**

Always log important trading events:
- ✅ Order placements
- ✅ Fills and partial fills
- ✅ Cancellations
- ✅ Merges
- ✅ Stop losses
- ✅ Arbitrage executions

### 3. **Include Context**

```python
# Good - includes context
logger.log_order(
    action='BUY',
    market=market_name,
    token=token_id,
    price=price,
    size=size,
    spread=spread,           # Extra context
    liquidity=liquidity      # Extra context
)

# Not as useful
logger.info("Order placed")
```

### 4. **Generate Daily Summaries**

```python
# At end of trading session or daily
logger.save_daily_summary()
logger.print_summary()  # Shows summary in console
```

### 5. **Monitor Log Size**

```bash
# Check log sizes
du -h logs/

# Clean old logs if needed
find logs/ -name "*.log.*" -mtime +30 -delete
```

## Troubleshooting

### Logs not being created

```python
# Check if logs directory exists
import os
if not os.path.exists('logs'):
    os.makedirs('logs')

# Initialize logger
logger = get_logger('test')
logger.info("Test message")
```

### Too much output

```python
# Reduce console verbosity
logger = get_logger('my_bot', console_level='WARNING')

# Or modify file level
logger = get_logger('my_bot', file_level='INFO')
```

### Parse errors in JSON logs

```python
# Validate JSON
import json

with open('logs/neg_risk_arb/neg_risk_arb_events.jsonl', 'r') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Error on line {i}: {e}")
```

## Migration Guide

### Converting Existing Bots

1. **Import the logger**
   ```python
   from poly_utils.logging_utils import get_logger
   ```

2. **Initialize at startup**
   ```python
   logger = get_logger('your_bot_name')
   ```

3. **Replace print statements**
   ```python
   # Before
   print(f"Placing order: {size} @ {price}")

   # After
   logger.info(f"Placing order: {size} @ {price}")
   ```

4. **Add trade logging**
   ```python
   # After creating orders
   logger.log_order(...)

   # After fills
   logger.log_fill(...)
   ```

5. **Add daily summaries**
   ```python
   # At end of session
   logger.save_daily_summary()
   ```

## Performance Impact

- Minimal CPU overhead (~1-2%)
- Log files grow predictably:
  - Detailed: ~1-5MB per day (active trading)
  - Trades: ~100KB per day
  - Events: ~50-200KB per day
- Rotation keeps disk usage bounded

## License

Same as parent project (MIT)
