# LLM Maker Bot

AI-aided market making bot that combines LLM strategic intelligence with fast execution.

## Overview

The LLM Maker Bot uses a two-loop architecture:

1. **Slow Loop (30-60s)**: LLM analyzes markets and generates trading signals
   - Evaluates order book imbalances
   - Detects price trends
   - Assesses portfolio risk
   - Provides directional bias and confidence scores

2. **Fast Loop (<1s)**: Execution engine places orders based on LLM signals
   - Event-driven (triggers on WebSocket price updates)
   - Respects LLM confidence thresholds
   - Implements risk checks and position limits
   - Falls back to pre-configured logic if LLM unavailable

## Architecture

```
llm_maker/
├── main.py              # Entry point, coordinates components
├── config.yaml          # Strategy and LLM configuration
├── market_analyzer.py   # Prepares market data for LLM
├── llm_strategy.py      # LLM integration and prompt generation
├── execution.py         # Fast order execution engine
├── logs/                # LLM decisions and execution logs
│   ├── llm_decisions.jsonl
│   └── executions.jsonl
└── positions/           # Risk management state files
```

## Key Features

### LLM Decision Making

The LLM analyzes each market and provides:
- **Action**: `buy_yes`, `buy_no`, `sell_yes`, `sell_no`, `hold`, `exit`
- **Confidence**: 0.0-1.0 score (higher = more confident)
- **Reasoning**: Brief explanation of the signal
- **Directional Bias**: `bullish`, `bearish`, `neutral`
- **Risk Level**: `low`, `medium`, `high`
- **Priority**: Which markets to focus on first

### Confidence-Based Position Sizing

- Base position size: 50 shares
- High confidence (>0.8): Up to 1.5x multiplier
- Low confidence (<0.6): No trade
- Max position size: 200 shares per side

### Directional Spread Adjustment

If LLM says market is bullish:
- Quote more aggressively on buy side (tighter spread)
- Quote less aggressively on sell side (wider spread)

### Fallback Logic

If LLM fails 3+ times consecutively:
- Automatically switches to rule-based trading
- Monitors stop-loss/take-profit thresholds
- Exits positions on predefined risk triggers
- Resumes LLM mode when service recovers

## Setup

### 1. Install Dependencies

```bash
# LLM SDK (choose based on provider in config.yaml)
pip install google-generativeai  # For Gemini (recommended)
# OR
pip install anthropic  # For Claude
# OR
pip install openai     # For OpenAI/DeepSeek
```

### 2. Configure Credentials

Add to `.env`:
```bash
# Trading account
LLM_MAKER_PK=your_private_key_here
LLM_MAKER_BROWSER_ADDRESS=your_wallet_address_here

# LLM API key (at least one required)
GEMINI_API_KEY=your_api_key_here  # Recommended: fast & cheap
# OR
ANTHROPIC_API_KEY=your_api_key_here
# OR
OPENAI_API_KEY=your_api_key_here
```

**Get API Keys:**
- Gemini: https://aistudio.google.com/apikey (Free tier: 15 RPM)
- Anthropic: https://console.anthropic.com/
- OpenAI: https://platform.openai.com/
- DeepSeek: https://platform.deepseek.com/ (Very cheap)

### 3. Configure Strategy

Edit `llm_maker/config.yaml`:

```yaml
llm:
  provider: "gemini"  # or "anthropic", "openai", "deepseek"
  max_requests_per_minute: 15

strategy:
  decision_interval: 45  # Seconds between LLM queries
  max_markets_per_query: 5
  min_confidence_to_trade: 0.6
  base_position_size: 50
  max_position_size: 200
```

### 4. Run the Bot

```bash
python llm_maker/main.py
```

## How It Works

### Market Analysis Flow

1. **Market Analyzer** selects top N markets based on:
   - Existing positions (high priority)
   - Low volatility (easier to make)
   - Tight spreads (efficient execution)

2. **LLM Strategy** generates prompt with:
   - Current portfolio state
   - Order book data (prices, spreads, imbalances)
   - Recent trends
   - Existing positions and P&L

3. **LLM** responds with JSON signals:
   ```json
   {
     "signals": [
       {
         "market_id": "0x123...",
         "action": "buy_yes",
         "confidence": 0.85,
         "reasoning": "Strong buying pressure, tight spread",
         "directional_bias": "bullish",
         "priority": "high"
       }
     ]
   }
   ```

4. **Execution Engine** places orders based on signals:
   - Checks confidence thresholds
   - Calculates position size
   - Adjusts spread based on directional bias
   - Implements risk checks

### Example LLM Decision

```
Market: "Will Trump win 2024?"
- YES Price: 0.65 | Spread: 0.02 | Imbalance: 1.3 (more buyers)
- Our Position: 50 shares @ 0.62 (P&L: +4.8%)

LLM Signal:
- Action: hold
- Confidence: 0.75
- Reasoning: "Position profitable, orderbook stable, hold for higher target"
- Bias: bullish
- Priority: medium

Execution: No action (holding), may adjust sell orders higher
```

## Configuration Options

### LLM Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `provider` | `gemini` | LLM provider to use |
| `max_requests_per_minute` | `15` | Rate limit (Gemini free tier) |
| `decision_interval` | `45` | Seconds between LLM queries |
| `max_markets_per_query` | `5` | Markets analyzed per query |

### Strategy Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_confidence_to_trade` | `0.6` | Minimum confidence to act |
| `min_confidence_to_size_up` | `0.8` | Confidence for larger positions |
| `base_position_size` | `50` | Base trade size |
| `max_position_size` | `200` | Maximum position per side |
| `stop_loss_percentage` | `-5.0` | Exit if down 5% |
| `take_profit_percentage` | `8.0` | Target 8% profit |

### Execution Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tick_improvement` | `1` | Ticks to improve over best |
| `max_spread_threshold` | `0.10` | Skip if spread > 10¢ |
| `min_liquidity_threshold` | `50` | Skip if liquidity < $50 |

## Logging and Monitoring

### LLM Decision Log

File: `llm_maker/logs/llm_decisions.jsonl`

Each line is a JSON object:
```json
{
  "timestamp": "2025-11-18T12:00:00",
  "model": "gemini-2.5-flash",
  "num_signals": 5,
  "signals": [...],
  "portfolio_state": {...}
}
```

### Execution Log

File: `llm_maker/logs/executions.jsonl`

Tracks all order placements and results.

### Viewing Logs

```bash
# Watch LLM decisions in real-time
tail -f llm_maker/logs/llm_decisions.jsonl | jq .

# Count signals by action
grep "buy_yes" llm_maker/logs/llm_decisions.jsonl | wc -l
```

## Cost Estimation

### API Costs (per 1000 decisions)

| Provider | Cost | Speed | Quality |
|----------|------|-------|---------|
| Gemini Flash | $0.02 | Fast | Good |
| DeepSeek | $0.01 | Medium | Good |
| Claude Haiku | $0.25 | Fast | Excellent |
| GPT-4o-mini | $0.15 | Fast | Excellent |

**Example**: Running 24/7 with 45s decision interval = ~2000 decisions/day
- Gemini: ~$0.04/day
- DeepSeek: ~$0.02/day
- Claude Haiku: ~$0.50/day

## Advantages vs Traditional Bots

| Feature | LLM Maker | OG Maker | Too Clever Maker |
|---------|-----------|----------|------------------|
| **Decision Logic** | AI-adaptive | Fixed rules | Complex rules |
| **Market Analysis** | Holistic (considers context) | Simple metrics | Multiple metrics |
| **Position Management** | Dynamic (confidence-based) | Simple thresholds | Defensive thresholds |
| **Trend Detection** | LLM interprets | None | Simple tracking |
| **Orderbook Analysis** | LLM evaluates imbalances | Basic ratio | Basic ratio |
| **Adaptability** | High (learns from data) | Low | Medium |
| **Explainability** | High (LLM provides reasoning) | N/A | N/A |

## When to Use LLM Maker

**Best For:**
- Markets with complex dynamics (not just simple mean reversion)
- Situations where context matters (news, trends, sentiment)
- When you want explanations for trading decisions
- Testing AI-driven strategies

**Not Ideal For:**
- Ultra-high frequency trading (LLM latency ~1-3s)
- Markets with extremely tight spreads (execution cost > edge)
- When API costs are a concern

## Troubleshooting

### LLM API Errors

```
Error: Missing API key: GEMINI_API_KEY not set
```
**Solution**: Add API key to `.env` file

```
Rate limit: 15/15 RPM. Waiting 45.2s...
```
**Solution**: Increase `decision_interval` or reduce `max_markets_per_query`

### Fallback Mode

```
WARNING: Switching to fallback mode after 3 consecutive failures
```
**Causes**:
- API quota exceeded
- Network issues
- Invalid API key

**Solution**: Check API status, verify key, wait for quota reset

### No Signals Generated

```
WARNING: No markets to analyze
```
**Causes**:
- All markets filtered out (volatility, spread thresholds)
- No markets in Google Sheets

**Solution**: Adjust `market_selection` filters in config.yaml

## Advanced Usage

### Custom Prompts

Edit `llm_strategy.py` → `_generate_prompt()` to customize:
- Analysis guidelines
- Response format
- Market context provided

### Multi-Model Ensemble

Run multiple LLM Maker instances with different providers:
- Terminal 1: Gemini (fast, cheap decisions)
- Terminal 2: Claude (high-quality decisions)
- Compare performance and combine insights

### Backtesting Signals

Collect decision logs for a week, then analyze:
```python
import json

with open('llm_maker/logs/llm_decisions.jsonl') as f:
    decisions = [json.loads(line) for line in f]

# Analyze signal accuracy
for decision in decisions:
    for signal in decision['signals']:
        # Compare signal to actual outcome
        # Calculate ROI if followed
```

## Future Enhancements

- [ ] Multi-timeframe analysis (LLM considers 1h, 4h, 24h trends)
- [ ] News integration (feed recent news to LLM for context)
- [ ] Portfolio-level optimization (LLM manages entire portfolio)
- [ ] Reinforcement learning (learn from outcomes)
- [ ] Ensemble voting (multiple LLMs vote on decisions)

## Support

For issues or questions:
1. Check logs: `llm_maker/logs/`
2. Review configuration: `llm_maker/config.yaml`
3. Test LLM connectivity: Run `poly_utils/llm_client.py` standalone
4. See main repo README for general setup

## License

Same as parent repository.
