# Spike Momentum Trading Bot

A sophisticated trading bot that identifies price spikes driven by real events by combining technical analysis, news monitoring, and LLM-powered context analysis.

## Core Strategy

**Thesis:** Catch price movements driven by real information events (news, game results, etc.) by distinguishing signal from noise.

**Key Innovation:** Use LLM to assess whether a price spike is justified by underlying events, with special focus on near-resolution markets (games ending, elections being called, etc.).

## Why This Works

Traditional spike trading has high false positive rates (noise, manipulation, whale moves). This bot adds context layers:

1. **Technical Layer**: Detect price spikes via WebSocket
2. **News Layer**: Fetch relevant news/events from multiple sources
3. **Intelligence Layer**: LLM analyzes whether spike is justified
4. **Near-Resolution Focus**: Prioritize markets approaching resolution for higher edge

## Current Status

✅ **Phase 1 Complete** - Basic Infrastructure
- Module structure created
- CLI framework (`scan`, `trade`, `auto` modes)
- Configuration system with YAML
- LLM provider abstraction (Gemini, Claude, OpenAI, OpenRouter)

🚧 **Phase 2 In Progress** - Core Components
- [ ] Spike detector
- [ ] News integration (sports focus)
- [ ] LLM analyzer with prompts
- [ ] Market scanner

📋 **Phase 3 Planned** - Trading & Risk
- [ ] Trade executor
- [ ] Position manager
- [ ] Risk controls

See [CHECKLIST.md](./CHECKLIST.md) for full implementation plan.

## Quick Start

### 1. Install Dependencies

```bash
# Choose your LLM provider and install SDK
pip install google-generativeai  # For Gemini (recommended)
# OR
pip install anthropic            # For Claude
# OR
pip install openai               # For GPT or OpenRouter
```

### 2. Create Configuration File

Copy the example configuration:

```bash
cp spike_momentum/config.yaml.example spike_momentum/config.yaml
```

### 3. Configure Environment

Add to your `.env` file:

```bash
# Bot account (use separate account for safety)
SPIKE_MOMENTUM_PK=your_private_key
SPIKE_MOMENTUM_BROWSER_ADDRESS=your_wallet_address

# LLM API key (choose one)
GEMINI_API_KEY=your_key_here      # Recommended: Fast + cheap
# ANTHROPIC_API_KEY=your_key_here # Alternative
# OPENAI_API_KEY=your_key_here    # Alternative
```

Get API keys:
- **Gemini**: https://aistudio.google.com/apikey (Free tier available!)
- Anthropic: https://console.anthropic.com/
- OpenAI: https://platform.openai.com/

**Note:** Your `config.yaml` file is gitignored for security. Never commit it to version control!

```yaml
llm:
  provider: "gemini"              # or "anthropic", "openai"
  min_confidence: 70              # 0-100 threshold

spike_detection:
  price_change_threshold: 0.02    # 2% minimum spike

position:
  size_pct: 0.05                  # 5% of account
  max_concurrent: 3
```

### 4. Run Scan Mode (Observation Only)

```bash
python -m spike_momentum.main scan
```

This will monitor markets and display detected spikes (no trading).

## LLM Provider Comparison

| Provider | Model | Speed | Cost (per 1M tokens) | Best For |
|----------|-------|-------|---------------------|----------|
| **Gemini** | 2.5 Flash | ⚡⚡⚡ | $0.075 | **Real-time** ⭐ |
| Gemini | 2.5 Flash-lite | ⚡⚡⚡⚡ | $0.0375 | Ultra high-frequency |
| Anthropic | Haiku 3.5 | ⚡⚡ | $0.25 | Complex reasoning |
| OpenAI | GPT-4o-mini | ⚡⚡ | $0.15 | Balanced |
| OpenRouter | Any model | Varies | Varies | Multi-model access |

**Recommendation:** Start with **Gemini 2.5 Flash** for optimal speed/cost balance.

## Operational Modes

### Scan Mode (Observation)
```bash
python -m spike_momentum.main scan
```
- Monitor markets for spikes
- Display LLM analysis
- No trading (safe learning mode)

### Trade Mode (Manual)
```bash
python -m spike_momentum.main trade --dry-run
```
- Detect spikes
- Analyze with LLM
- Prompt for confirmation
- Execute trades

### Auto Mode (Automated)
```bash
python -m spike_momentum.main auto --dry-run
```
- Fully automated operation
- Continuous monitoring
- Auto-execution based on rules

**Note:** Always start with `--dry-run` to paper trade first!

## Strategy Configuration

### Spike Detection
```yaml
spike_detection:
  price_change_threshold: 0.02      # 2% minimum
  time_windows: [30, 60, 300]       # Track 30s, 1min, 5min
  volume_threshold: 100             # Min $ volume
```

### LLM Analysis
```yaml
llm:
  enabled: true
  provider: "gemini"
  min_confidence: 70                # 0-100 threshold
  required_for_trade: true          # Must have LLM approval
```

### Near-Resolution Boost
```yaml
near_resolution:
  enabled: true
  time_threshold_hours: 4           # Market resolves soon
  boost_factor: 1.5                 # Increase confidence
  sports:
    in_game_boost: 2.0              # Active games
```

### Risk Controls
```yaml
risk:
  max_daily_loss: -100              # $ limit
  max_daily_trades: 20
  consecutive_loss_limit: 3         # Pause after losses
```

## News Sources (Sports Focus)

The bot will integrate these news sources (in priority order):

1. **ESPN RSS** - Free, reliable sports news
2. **TheScore RSS** - Real-time scores
3. **Google News RSS** - General catch-all
4. **(Future) SportsData.io API** - Paid, real-time data

## Example Flow

```
1. [9:45 PM] Spike detected: "Lakers to win" 0.65 → 0.78 (+20%)

2. [9:45 PM] News fetched:
   - ESPN: "Lakers lead 98-89, 2:14 remaining in 4th"
   - Twitter: "LeBron hits back-to-back 3-pointers"

3. [9:45 PM] LLM analyzes (Gemini Flash):
   "9-point lead with 2 min left = ~85% win probability.
    Price of 0.78 is reasonable. Confidence: 85"

4. [9:45 PM] Decision: TRADE
   - Enter: 0.78
   - Target: 0.95
   - Stop: 0.70
   - Max hold: 10 min

5. [9:48 PM] Game ends, Lakers win → 1.00
   Profit: +22% in 3 minutes ✓
```

## Architecture

```
WebSocket (Price Data)
       ↓
Spike Detector → Quality Filters
       ↓
News Monitor (Sports APIs)
       ↓
LLM Analyzer (Gemini/Claude/GPT)
       ↓
Signal Fusion
       ↓
Trade Executor → Risk Manager
```

## Safety Features

- **Dry-run mode**: Paper trade without risk
- **Position limits**: Max concurrent positions
- **Stop-loss**: Automatic exit on loss
- **Time limits**: Max holding period
- **Rate limits**: Control LLM API costs
- **Circuit breakers**: Pause on consecutive losses

## Cost Estimation

**LLM API Costs:**
- Gemini Flash: ~$0.01-0.05 per day (20-50 analyses)
- News APIs: $0-50/month (free tiers available)

**Total:** ~$50/month infrastructure cost

If the bot makes even $2/day profit, it pays for itself many times over.

## Development Status

This is an **experimental bot in development**. Key components still being built:

- ✅ LLM provider abstraction
- ✅ Configuration system
- ✅ CLI framework
- 🚧 Spike detector
- 🚧 News integration
- 🚧 LLM analyzer
- 📋 Trade executor
- 📋 Risk manager

## Next Steps

1. **Review config**: Edit `config.yaml` for your preferences
2. **Get API key**: Sign up for Gemini AI Studio (free tier)
3. **Test LLM**: Verify API key works
4. **Wait for completion**: Core components being built
5. **Run scan mode**: Observe spikes without trading
6. **Paper trade**: Test with `--dry-run`
7. **Go live**: Start with minimal capital

## Contributing

See [CHECKLIST.md](./CHECKLIST.md) for the full implementation roadmap.

## Risks & Disclaimers

⚠️ **This is experimental software**
- Test with small amounts only
- May lose money
- LLM analysis not perfect
- False signals possible
- Use at your own risk

Start conservatively, monitor closely, iterate based on results.

## Support

For questions or issues, refer to:
- [CHECKLIST.md](./CHECKLIST.md) - Full implementation plan
- Main repo README - General setup instructions
- Configuration comments in `config.yaml`

---

**Current Version:** 0.1.0 (Development)
**Status:** Phase 1 Complete, Phase 2 In Progress
**Last Updated:** 2025-11-17
