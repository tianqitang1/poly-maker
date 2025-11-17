# Spike Momentum Bot - Implementation Checklist

## Strategy Overview

**Core Thesis:** Catch price movements driven by real information events (news, game results, etc.) by combining:
1. Real-time price spike detection
2. News/event monitoring
3. LLM-powered context analysis to distinguish signal from noise

**Key Innovation:** Use LLM to assess whether a price spike is justified by underlying events, focusing on near-resolution markets.

---

## Phase 1: Basic Infrastructure ✓ Setup

### 1.1 Module Structure
- [ ] Create `spike_momentum/` directory
- [ ] Create `__init__.py`
- [ ] Create `main.py` with CLI structure
- [ ] Create `config.yaml.example` with base parameters (config.yaml gitignored)
- [ ] Add environment variables to `.env.example`:
  - `SPIKE_MOMENTUM_PK`
  - `SPIKE_MOMENTUM_BROWSER_ADDRESS`
  - News API keys (to be determined)

### 1.2 Configuration System
- [ ] Define `config.yaml.example` schema
- [ ] Implement config loading in Python
- [ ] Add validation for required parameters
- [ ] Support for live vs. dry-run modes

---

## Phase 2: Price Spike Detection (Technical Layer)

### 2.1 Spike Detector (`spike_detector.py`)
- [ ] Integrate with existing WebSocket handlers (`poly_data/websocket_handlers.py`)
- [ ] Implement price change tracking over multiple time windows:
  - [ ] 30-second window
  - [ ] 1-minute window
  - [ ] 5-minute window
- [ ] Calculate price velocity (rate of change)
- [ ] Track volume accompanying price changes
- [ ] Monitor bid-ask spread changes
- [ ] Detect order book depth changes
- [ ] Calculate volatility baseline (rolling average)
- [ ] Anomaly detection: spike significance vs. baseline

### 2.2 Quality Filters
- [ ] Minimum liquidity threshold (filter thin books)
- [ ] Volume confirmation (real trades, not just spread changes)
- [ ] Market size filters (avoid manipulation-prone markets)
- [ ] Spread consistency check (detect wash trading)
- [ ] Time-of-day filters (avoid low-liquidity hours)

### 2.3 Market Scanner (`momentum_scanner.py`)
- [ ] Scan all active markets for spikes
- [ ] Rank opportunities by quality score
- [ ] Track market metadata (category, resolution date, etc.)
- [ ] Maintain blacklist/whitelist
- [ ] Filter by market characteristics

---

## Phase 3: News & Event Monitoring (Context Layer) ⭐ NEW

### 3.1 News Source Integration (`news_monitor.py`)
**Goal:** Detect real-world events that could cause legitimate price movements

#### News APIs to Consider:
- [ ] **Google News RSS** - Free, easy to implement
  - [ ] Set up RSS feed parser
  - [ ] Filter by categories (sports, politics, business)
  - [ ] Extract timestamps and headlines
- [ ] **NewsAPI.org** - Comprehensive news aggregation
  - [ ] Sign up for API key
  - [ ] Implement rate-limited fetching
  - [ ] Track recent articles by keyword
- [ ] **Twitter/X API** - Real-time breaking news
  - [ ] Monitor specific accounts (e.g., @AP, @Reuters, sports reporters)
  - [ ] Track hashtags related to active markets
  - [ ] Parse tweets for relevant keywords
- [ ] **Sports-specific sources:**
  - [ ] ESPN API (if available)
  - [ ] TheScore API
  - [ ] Live game score feeds
  - [ ] In-game event streams (goals, touchdowns, etc.)
- [ ] **Election/Politics sources:**
  - [ ] Decision Desk HQ
  - [ ] Associated Press election results
  - [ ] Polymarket's own activity feed
  - [ ] FiveThirtyEight updates

#### Implementation:
- [ ] Create unified news event schema
- [ ] Build news aggregator with deduplication
- [ ] Implement keyword matching to markets
- [ ] Track event timestamps for freshness
- [ ] Cache recent news (last 1-6 hours)
- [ ] Rate limiting and error handling

### 3.2 Market-News Matching (`context_matcher.py`)
- [ ] Extract keywords from market questions
- [ ] Match news articles to relevant markets
- [ ] Score relevance (exact match vs. partial match)
- [ ] Track recency (prefer fresh news)
- [ ] Build market context database

### 3.3 Event Detection
- [ ] Parse sports scores and game status
- [ ] Track election result updates
- [ ] Monitor court decisions / regulatory announcements
- [ ] Identify market-moving events in real-time

---

## Phase 4: LLM-Powered Analysis (Intelligence Layer) ⭐ NEW

### 4.1 Context Analyzer (`llm_analyzer.py`)
**Goal:** Use LLM to assess whether price spike is justified by news/events

#### Core Functionality:
- [ ] Implement LLM integration (Claude API or OpenAI)
- [ ] Choose model for speed/cost balance:
  - Claude Haiku (fast, cheap, good reasoning)
  - GPT-4o-mini (fast, cheap)
  - Sonnet 4.5 (more expensive but better reasoning)
- [ ] Build prompt template for spike analysis
- [ ] Implement result parsing and scoring

#### Prompt Structure:
```
Market Question: "{market_question}"
Current Price: {current_price}
Price 5min ago: {price_5min_ago}
Price Change: +{pct_change}%

Recent News:
{news_headlines_and_snippets}

Analysis Task:
1. Does this news directly relate to the market question?
2. Does the news support the price movement direction?
3. Is this information likely already priced in?
4. What is the confidence level (0-100) that this spike is justified?
5. Is the market near resolution (outcome becoming clear)?

Respond in JSON format.
```

#### Decision Logic:
- [ ] Parse LLM response (JSON format)
- [ ] Extract confidence score
- [ ] Extract reasoning/justification
- [ ] Combine with technical signals
- [ ] Generate final trade decision

### 4.2 Signal Fusion (`signal_fusion.py`)
**Combine technical + news + LLM signals**

- [ ] Weight different signal types:
  - Technical spike strength (0-1)
  - News relevance score (0-1)
  - LLM confidence score (0-1)
  - Time-to-resolution factor (0-1)
- [ ] Calculate composite opportunity score
- [ ] Apply minimum thresholds for each signal type
- [ ] Require multiple signals to align
- [ ] Handle conflicting signals (e.g., spike up but negative news)

### 4.3 Near-Resolution Detection ⭐ YOUR KEY INSIGHT
**Identify markets approaching resolution**

- [ ] Parse market resolution conditions
- [ ] Detect time-based triggers:
  - [ ] Games in 4th quarter / final period
  - [ ] Elections with >90% reporting
  - [ ] Event dates within 24-48 hours
- [ ] Calculate "time to resolution" metric
- [ ] Boost confidence for near-resolution + news combination
- [ ] Special handling for different market types:
  - [ ] Sports: Track game time, score differential
  - [ ] Elections: Track % reporting, margin
  - [ ] Binary events: Track deadline proximity

---

## Phase 5: Execution & Position Management

### 5.1 Trade Executor (`executor.py`)
- [ ] Integrate with `PolymarketClient`
- [ ] Implement order placement (market vs. limit)
- [ ] Handle slippage and partial fills
- [ ] Position size calculation based on:
  - [ ] Account balance
  - [ ] Signal strength
  - [ ] Market liquidity
  - [ ] Risk limits
- [ ] Order confirmation and tracking
- [ ] Failed order retry logic

### 5.2 Position Manager
- [ ] Track open positions
- [ ] Monitor P&L in real-time
- [ ] Associate positions with entry signals
- [ ] Store entry context (news, LLM analysis) for later review

### 5.3 Exit Manager
- [ ] Implement stop-loss logic
- [ ] Implement take-profit logic
- [ ] Time-based exit (max hold period)
- [ ] Trailing stop implementation
- [ ] Reversal detection (exit on opposing spike)
- [ ] Event-driven exit (e.g., market resolved, contradicting news)
- [ ] LLM-assisted exit decision (optional):
  - Query: "Given new information, should we hold or exit?"

---

## Phase 6: Risk Management

### 6.1 Risk Manager (`risk_manager.py`)
- [ ] Position sizing rules
- [ ] Maximum concurrent positions
- [ ] Per-trade risk limit (% of account)
- [ ] Daily loss limit
- [ ] Drawdown monitoring
- [ ] Circuit breakers:
  - [ ] Consecutive loss limit
  - [ ] Daily trade limit
  - [ ] Cooldown periods after losses

### 6.2 Market Selection Risk
- [ ] Blacklist manipulated markets
- [ ] Whitelist high-quality markets
- [ ] Minimum liquidity requirements
- [ ] Maximum position per market
- [ ] Category diversification (avoid overexposure)

### 6.3 LLM Risk Controls ⭐ NEW
- [ ] API rate limiting (cost control)
- [ ] Response timeout handling
- [ ] Fallback logic if LLM unavailable
- [ ] Cache LLM responses for same events
- [ ] Monitor LLM accuracy over time
- [ ] Human override capability

---

## Phase 7: Operational Modes

### 7.1 Scan Mode
```bash
python -m spike_momentum.main scan
```
- [ ] Monitor markets in real-time
- [ ] Display detected spikes with context
- [ ] Show news items and LLM analysis
- [ ] No trading, observation only
- [ ] Log all signals for backtesting

### 7.2 Manual Trade Mode
```bash
python -m spike_momentum.main trade --dry-run
```
- [ ] Detect opportunity
- [ ] Fetch relevant news
- [ ] Run LLM analysis
- [ ] Present full context to user
- [ ] User confirms/rejects
- [ ] Execute trade
- [ ] Monitor position

### 7.3 Auto Mode
```bash
python -m spike_momentum.main auto
```
- [ ] Fully automated operation
- [ ] Continuous spike monitoring
- [ ] Automatic news fetching
- [ ] LLM analysis without user input
- [ ] Auto-execution based on thresholds
- [ ] Real-time risk management
- [ ] Performance reporting

### 7.4 Backtest Mode (Future)
```bash
python -m spike_momentum.main backtest --data historical.csv
```
- [ ] Replay historical price data
- [ ] Simulate news events (if available)
- [ ] Test parameter combinations
- [ ] Generate performance metrics
- [ ] Optimize signal thresholds

---

## Phase 8: Analytics & Monitoring

### 8.1 Performance Tracking (`analytics.py`)
- [ ] Win rate overall
- [ ] Win rate by signal type:
  - [ ] Technical-only signals
  - [ ] News-confirmed signals
  - [ ] LLM high-confidence signals
  - [ ] Near-resolution signals
- [ ] Average hold time (winners vs. losers)
- [ ] Best performing market categories
- [ ] Time-of-day performance
- [ ] News source effectiveness
- [ ] LLM accuracy tracking

### 8.2 Signal Quality Metrics
- [ ] False positive rate (spike without follow-through)
- [ ] Missed opportunities (spike with news but no trade)
- [ ] LLM confidence vs. outcome correlation
- [ ] News latency (how old was news when we acted)
- [ ] Time-to-resolution vs. success rate

### 8.3 Reporting & Alerts
- [ ] Daily performance summary
- [ ] Risk limit violations
- [ ] Unusual patterns (good or bad)
- [ ] LLM API errors or slowness
- [ ] News feed failures
- [ ] Trade execution issues

### 8.4 Logging
- [ ] Comprehensive signal logs (why we entered)
- [ ] News snapshots at entry time
- [ ] LLM reasoning for each trade
- [ ] Exit reasons and outcomes
- [ ] Structured logs for later analysis

---

## Phase 9: Testing & Validation

### 9.1 Unit Tests
- [ ] Spike detector logic
- [ ] News matching algorithm
- [ ] LLM prompt/response parsing
- [ ] Signal fusion calculations
- [ ] Risk limit enforcement

### 9.2 Integration Tests
- [ ] WebSocket data flow
- [ ] News API integration
- [ ] LLM API integration
- [ ] Order execution pipeline
- [ ] Full scan-to-trade workflow

### 9.3 Paper Trading
- [ ] Run in dry-run mode for 1-2 weeks
- [ ] Track hypothetical performance
- [ ] Identify edge cases
- [ ] Tune parameters
- [ ] Validate risk controls

### 9.4 Small-Scale Live Testing
- [ ] Start with minimal capital ($50-100)
- [ ] Monitor closely
- [ ] Iterate on parameters
- [ ] Expand gradually if successful

---

## Phase 10: Documentation

### 10.1 User Documentation
- [ ] README.md for spike_momentum module
- [ ] Configuration guide
- [ ] API key setup instructions
- [ ] Usage examples
- [ ] Troubleshooting guide

### 10.2 Developer Documentation
- [ ] Architecture overview
- [ ] Signal flow diagram
- [ ] LLM prompt engineering notes
- [ ] News source integration guide
- [ ] Extension points for new signals

### 10.3 Strategy Documentation
- [ ] Parameter tuning guide
- [ ] Market selection criteria
- [ ] Known limitations
- [ ] Performance expectations
- [ ] Risk management rules

---

## Configuration Parameters (config.yaml.example)

**Note:** Copy `config.yaml.example` to `config.yaml` and customize. The `config.yaml` file is gitignored for security.

### Technical Spike Detection
```yaml
spike_detection:
  price_change_threshold: 0.02      # 2% minimum price move
  time_windows: [30, 60, 300]       # seconds
  volume_threshold: 100             # minimum $ volume
  min_liquidity: 500                # minimum order book depth
  volatility_multiplier: 2.0        # spike must be 2x normal volatility
```

### News Monitoring
```yaml
news:
  enabled: true
  sources:
    - google_news_rss
    - newsapi_org
    - twitter
  refresh_interval: 60              # seconds
  max_age: 3600                     # only consider news from last hour
  relevance_threshold: 0.6          # 0-1 score
  required_for_trade: false         # can trade without news (use as filter)
```

### LLM Analysis
```yaml
llm:
  enabled: true
  provider: "anthropic"             # or "openai"
  model: "claude-haiku-3-5"         # fast & cheap
  # model: "claude-sonnet-4-5"      # better reasoning (more expensive)
  timeout: 10                       # seconds
  min_confidence: 70                # 0-100 threshold for trading
  required_for_trade: true          # must have LLM approval
  cache_responses: true             # avoid re-analyzing same event
```

### Signal Fusion
```yaml
signal_fusion:
  weights:
    technical: 0.3                  # spike strength
    news: 0.3                       # news relevance
    llm: 0.4                        # LLM confidence
  min_composite_score: 0.7          # 0-1 threshold
  require_all_signals: false        # OR vs AND logic
```

### Near-Resolution Boost
```yaml
near_resolution:
  enabled: true
  time_threshold_hours: 4           # market resolves within 4 hours
  boost_factor: 1.5                 # multiply confidence by this
  special_categories:
    sports:
      in_game_boost: 2.0            # active games get extra boost
    elections:
      reporting_threshold: 0.75     # >75% reporting
```

### Position Management
```yaml
position:
  size_pct: 0.05                    # 5% of account
  max_concurrent: 3
  max_position_value: 500

  # Dynamic sizing based on signal quality
  size_by_confidence:
    enabled: true
    min_size_pct: 0.02              # at min confidence
    max_size_pct: 0.10              # at 100% confidence
```

### Exit Rules
```yaml
exits:
  take_profit_pct: 0.03
  stop_loss_pct: -0.025
  max_hold_time_sec: 900            # 15 minutes
  trailing_stop: true
  trailing_offset_pct: 0.015

  # Event-driven exits
  exit_on_contradicting_news: true
  exit_on_market_resolved: true
  llm_assisted_exit: false          # optional: ask LLM about exits
```

### Risk Controls
```yaml
risk:
  max_daily_loss: -100
  max_daily_trades: 20
  consecutive_loss_limit: 3
  cooldown_after_loss_sec: 300

  # LLM-specific limits
  max_llm_calls_per_hour: 100       # cost control
  max_llm_cost_per_day: 10          # $ limit
```

---

## Success Metrics

### Phase 1-2 (Technical Foundation)
- [ ] Successfully detect 10+ spikes per day
- [ ] <1% false positives (spikes in low-liquidity markets)
- [ ] Spike detection latency < 5 seconds

### Phase 3 (News Integration)
- [ ] Successfully match news to markets 80%+ of time
- [ ] News latency < 2 minutes from publication
- [ ] Zero downtime on news feeds

### Phase 4 (LLM Analysis)
- [ ] LLM response time < 5 seconds
- [ ] LLM confidence correlates with trade success (>0.6 correlation)
- [ ] Cost < $5 per day in API fees

### Phase 9 (Live Trading)
- [ ] Win rate > 55%
- [ ] Profit factor > 1.5
- [ ] Max drawdown < 15%
- [ ] Sharpe ratio > 1.0 (if enough trades)

---

## Timeline Estimate

- **Week 1:** Phase 1-2 (Basic spike detection) - Can start observing
- **Week 2:** Phase 3 (News integration) - Major value add
- **Week 3:** Phase 4 (LLM analysis) - The secret sauce
- **Week 4:** Phase 5-6 (Execution + Risk) - Ready to paper trade
- **Week 5:** Phase 9 (Testing) - Validate before live
- **Week 6:** Small-scale live deployment

---

## Key Decision Points

### Decision 1: LLM Required vs. Optional
**Question:** Should LLM approval be required for every trade, or just used as an additional filter?

**Option A (Required):** More conservative, fewer trades, higher quality
**Option B (Optional):** More trades, some without context, faster execution

**Recommendation:** Start with Required, loosen if missing good opportunities

### Decision 2: News Sources Priority
**Question:** Which news sources to implement first?

**Recommendation:**
1. **Sports:** Easy to test, clear events, fast feedback
   - Start with TheScore or ESPN RSS
   - Focus on major leagues (NFL, NBA, soccer)
2. **Politics:** Good for elections
   - Associated Press, Decision Desk HQ
3. **General:** Catch unexpected events
   - Google News RSS (free, easy)

### Decision 3: LLM Model Selection
**Question:** Which model for best speed/cost/accuracy balance?

**Options:**
- Claude Haiku 3.5: Fast (~1s), cheap ($0.25/M tokens), good reasoning
- GPT-4o-mini: Fast, cheap, decent
- Claude Sonnet 4.5: Best reasoning, slower, more expensive

**Recommendation:** Start with Haiku 3.5, upgrade to Sonnet if accuracy issues

### Decision 4: Signal Combination Logic
**Question:** Require all signals (AND) or any strong signal (OR)?

**Recommendation:**
- Require: (Spike detected) AND (News exists OR Near-resolution) AND (LLM confidence > threshold)
- This ensures we have context for every trade

---

## Risk Assessment

### Technical Risks
- **LLM latency:** May miss fast-moving opportunities
  - Mitigation: Use fast model, set timeout, have fallback
- **News feed reliability:** API outages, rate limits
  - Mitigation: Multiple sources, graceful degradation
- **API costs:** LLM calls can add up
  - Mitigation: Caching, rate limits, daily budget caps

### Strategy Risks
- **LLM hallucination:** Model could misinterpret news
  - Mitigation: Require structured JSON output, validate responses
- **Stale news:** Acting on old information
  - Mitigation: Strict recency filters, timestamp checking
- **False confidence:** LLM confident about wrong conclusion
  - Mitigation: Track LLM accuracy, require multiple signals, position limits

### Market Risks
- **Still being exit liquidity:** Even with news, could be too late
  - Mitigation: Focus on near-resolution markets, strict time limits
- **Manipulation:** Fake news or coordinated price moves
  - Mitigation: Verify news sources, volume confirmation, position limits
- **Whipsaw:** Momentum reverses quickly
  - Mitigation: Tight stops, trailing stops, max hold time

---

## Next Steps

1. **Review & Approve:** Does this approach match your vision?
2. **Prioritize:** Which phases are most important to you?
3. **News Sources:** Which categories do you want to focus on? (sports, politics, other?)
4. **LLM Provider:** Anthropic (Claude) or OpenAI (GPT)?
5. **Start Building:** Begin with Phase 1-2 infrastructure

---

## Notes & Ideas

- Could use this same infrastructure for other strategies (not just momentum)
- LLM could also help with market selection (scan all markets, ask "which are near resolution?")
- Could build a "news replay" backtesting system
- Integration with your existing Google Sheets for manual overrides
- Could add Telegram/Discord alerts for high-confidence opportunities
- Consider building a simple web dashboard to visualize signals in real-time

---

**Last Updated:** 2025-11-17
**Status:** Planning Phase
**Owner:** TBD
