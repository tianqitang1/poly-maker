# Poly Utils - Shared Utilities for Poly-Maker Bots

This package provides generalized, reusable utilities for all trading bots in the poly-maker suite.

## Features

- **LLM Client**: Unified interface for multiple LLM providers (Gemini, Claude, GPT, etc.)
- **News Feed**: Multi-category news aggregator (sports, crypto, politics, general)
- **Logging**: Structured logging system
- **Proxy Config**: Proxy management
- **Google Utils**: Google Sheets integration

---

## 🤖 LLM Client

A unified interface for querying multiple LLM providers with built-in rate limiting, retry logic, and cost tracking.

### Supported Providers

| Provider | Model | Recommended For | Cost |
|----------|-------|----------------|------|
| **Gemini** | gemini-2.5-flash | Speed + cost balance | Very Low |
| **DeepSeek** | deepseek-chat | High-frequency queries | Very Low |
| **OpenAI** | gpt-4o-mini | General purpose | Low |
| **Anthropic** | claude-haiku-3-5 | High-quality analysis | Medium |
| **OpenRouter** | Any model | Multi-model access | Varies |

### Basic Usage

```python
from poly_utils import LLMClient
import yaml

# Load config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Initialize client
llm = LLMClient(config['llm'], bot_name='my_bot')

# Query with text response
response = llm.query("Analyze this market: Will Bitcoin reach $100k?")
if response['success']:
    analysis = response['content']
    print(analysis)

# Query with JSON response
response = llm.query(
    "Analyze this market. Respond with JSON: {\"confidence\": 0-100, \"reasoning\": \"...\"}",
    json_mode=True
)
if response['success']:
    data = response['content']  # Already parsed as dict
    print(f"Confidence: {data['confidence']}%")

# Get usage stats
stats = llm.get_stats()
print(f"Total calls: {stats['total_calls']}")
print(f"Estimated cost: ${stats['estimated_cost']:.2f}")
print(f"Current RPM: {stats['rpm_current']}/{stats['rpm_limit']}")
```

### Configuration

```yaml
llm:
  enabled: true
  provider: "gemini"  # or anthropic, openai, openrouter, deepseek

  # Rate limits
  max_requests_per_minute: 15
  max_calls_per_hour: 100
  max_cost_per_day: 10.0

  # Provider settings
  gemini:
    model: "gemini-2.5-flash"
    api_key_env: "GEMINI_API_KEY"
```

### Advanced Usage

```python
# Custom temperature and max tokens
response = llm.query(
    prompt="Your prompt here",
    json_mode=True,
    temperature=0.7,      # 0-1 (higher = more creative)
    max_tokens=2048       # Maximum response length
)

# Custom retry logic
response = llm.query(
    prompt="Your prompt here",
    retries=5  # Override default retry count
)
```

### API Keys Setup

Set environment variables for your chosen provider:

```bash
export GEMINI_API_KEY="your_key_here"
export ANTHROPIC_API_KEY="your_key_here"
export OPENAI_API_KEY="your_key_here"
export OPENROUTER_API_KEY="your_key_here"
export DEEPSEEK_API_KEY="your_key_here"
```

---

## 📰 News Feed

A generalized news aggregator supporting multiple categories and sources with smart market matching.

### Supported Categories

- **Sports**: ESPN, TheScore (NFL, NBA, MLB, NHL, Soccer, etc.)
- **Crypto**: CoinDesk, CoinTelegraph, Bitcoin Magazine, Decrypt
- **Politics**: Politico, The Hill, RealClearPolitics
- **General**: Reuters, AP News

### Basic Usage

```python
from poly_utils import NewsFeed, NewsItem
import yaml

# Load config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Initialize feed
feed = NewsFeed(config['news'])

# Fetch all news
all_news = feed.fetch_news(max_items=50)
for item in all_news:
    print(f"[{item.category}] {item.title}")

# Fetch category-specific news
sports_news = feed.fetch_news(category='sports', max_items=20)
crypto_news = feed.fetch_news(category='crypto', max_items=20)

# Search by keywords
bitcoin_news = feed.search_news(
    keywords=['bitcoin', 'btc'],
    category='crypto',
    max_results=10
)

# Match news to market question (smart relevance matching)
market_question = "Will the Lakers beat the Warriors tonight?"
matches = feed.match_to_market(market_question, max_results=5)

for match in matches:
    news_item = match['news']
    score = match['relevance_score']
    keywords = match['matched_keywords']

    print(f"[{score:.2f}] {news_item.title}")
    print(f"Keywords: {', '.join(keywords)}")
```

### Configuration

```yaml
news:
  enabled: true
  refresh_interval: 60    # Seconds between fetches
  max_age: 3600          # Max age of news items (1 hour)

  categories:
    - sports
    - crypto

  sources:
    espn_rss:
      enabled: true
      leagues: [nfl, nba, mlb]

    coindesk_rss:
      enabled: true
```

### NewsItem Object

```python
item = NewsItem(
    title="Lakers win 108-102",
    summary="LeBron James scores 30 points...",
    link="https://espn.com/example",
    published=datetime.now(),
    source="ESPN",
    category="sports",
    subcategory="NBA",
    tags=["lakers", "warriors"]
)

# Convert to dict
data = item.to_dict()
# {
#   'title': '...',
#   'summary': '...',
#   'age_seconds': 120,
#   ...
# }
```

---

## 🔗 Combined Usage (LLM + News)

The real power comes from combining LLM analysis with news context:

```python
from poly_utils import LLMClient, NewsFeed
import yaml

# Load config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Initialize both
llm = LLMClient(config['llm'], bot_name='my_bot')
feed = NewsFeed(config['news'])

# Market question
market_question = "Will Bitcoin reach $100k by end of month?"

# Get relevant news
news_matches = feed.match_to_market(market_question, max_results=5)

# Build context for LLM
news_context = "\n".join([
    f"- [{item['news'].source}] {item['news'].title}"
    for item in news_matches
])

# Query LLM with news context
prompt = f"""
Analyze this prediction market:

QUESTION: {market_question}

RECENT NEWS:
{news_context}

Provide your analysis in JSON format:
{{
  "justified": true/false,
  "confidence": 0-100,
  "reasoning": "...",
  "recommendation": "buy/sell/hold"
}}
"""

response = llm.query(prompt, json_mode=True)
if response['success']:
    analysis = response['content']
    print(f"Confidence: {analysis['confidence']}%")
    print(f"Recommendation: {analysis['recommendation']}")
    print(f"Reasoning: {analysis['reasoning']}")
```

---

## 🎯 Bot-Specific Examples

### Spike Momentum (Sports)

```yaml
llm:
  enabled: true
  provider: "gemini"
  min_confidence: 70

news:
  enabled: true
  categories: ["sports"]
  sources:
    espn_rss: {enabled: true, leagues: [nfl, nba]}
    thescore_rss: {enabled: true, leagues: [nfl, nba]}
```

```python
# Detect spike, get news, analyze with LLM
spike_detected = detect_price_spike(market)
if spike_detected:
    news = feed.match_to_market(market.question, category='sports')
    analysis = llm.query(build_spike_prompt(market, news), json_mode=True)

    if analysis['content']['justified']:
        execute_trade(market, analysis['content']['recommendation'])
```

### Near-Sure (Market Validation)

```yaml
llm:
  enabled: true
  provider: "gemini"

news:
  enabled: true
  categories: ["sports", "politics"]
```

```python
# Validate near-certain markets with news
for market in near_certain_markets:
    news = feed.match_to_market(market.question)

    prompt = f"""
    This market is priced at {market.price:.2f} (near certain).

    Recent news: {format_news(news)}

    Is this market truly resolved or is it manipulation?
    JSON: {{"validated": true/false, "reasoning": "..."}}
    """

    response = llm.query(prompt, json_mode=True)
    if response['content']['validated']:
        trade(market)
```

### Neg Risk Arb (Pair Validation)

```yaml
llm:
  enabled: true
  provider: "deepseek"  # Cheap for validation
```

```python
# Validate arbitrage pairs are truly complementary
for arb_opportunity in opportunities:
    prompt = f"""
    Market A: {arb_opportunity.market_a.question}
    Market B: {arb_opportunity.market_b.question}

    Are these truly binary opposites that resolve from the same event?
    JSON: {{"valid_pair": true/false, "reasoning": "..."}}
    """

    response = llm.query(prompt, json_mode=True)
    if response['content']['valid_pair']:
        execute_arbitrage(arb_opportunity)
```

### Market Maker (Volatility Prediction)

```yaml
llm:
  enabled: true
  provider: "gemini"
  max_requests_per_minute: 30

news:
  enabled: true
  categories: ["sports", "crypto", "politics"]
  refresh_interval: 30
```

```python
# Adjust spreads based on predicted volatility
recent_news = feed.get_recent_news(max_age_seconds=300)  # Last 5 minutes

for market in active_markets:
    relevant_news = feed.match_to_market(market.question)

    if relevant_news:
        prompt = f"""
        Market: {market.question}
        Current price: {market.price}

        Recent news: {format_news(relevant_news)}

        Predict volatility in next 5 minutes.
        JSON: {{"volatility": "low/medium/high", "reasoning": "..."}}
        """

        response = llm.query(prompt, json_mode=True)
        volatility = response['content']['volatility']

        # Widen spreads for high volatility
        if volatility == 'high':
            adjust_spread(market, multiplier=2.0)
```

---

## 📊 Cost Estimation

### LLM Costs (per 1,000 queries)

| Provider | Model | Estimated Cost |
|----------|-------|----------------|
| Gemini | gemini-2.5-flash | $0.10 - $0.50 |
| DeepSeek | deepseek-chat | $0.05 - $0.20 |
| OpenAI | gpt-4o-mini | $0.50 - $2.00 |
| Anthropic | claude-haiku-3-5 | $1.00 - $3.00 |

### News Feed Costs

All RSS feeds are **free** and require no API keys.

---

## 🔧 Configuration Reference

See `config.example.yaml` for a complete configuration template.

### Minimal Config (Sports Only)

```yaml
llm:
  enabled: true
  provider: "gemini"
  gemini:
    model: "gemini-2.5-flash"
    api_key_env: "GEMINI_API_KEY"

news:
  enabled: true
  categories: ["sports"]
  sources:
    espn_rss: {enabled: true, leagues: [nfl, nba]}
```

### Full Config (All Features)

```yaml
llm:
  enabled: true
  provider: "gemini"
  max_requests_per_minute: 15
  max_calls_per_hour: 100
  max_cost_per_day: 10.0
  max_retries: 2
  timeout: 10
  gemini:
    model: "gemini-2.5-flash"
    api_key_env: "GEMINI_API_KEY"

news:
  enabled: true
  refresh_interval: 60
  max_age: 3600
  categories: ["sports", "crypto", "politics"]
  sources:
    espn_rss: {enabled: true, leagues: [nfl, nba, mlb, nhl]}
    thescore_rss: {enabled: true, leagues: [nfl, nba]}
    coindesk_rss: {enabled: true}
    politico_rss: {enabled: true}
```

---

## 🚀 Getting Started

1. **Install dependencies**:
   ```bash
   pip install google-generativeai anthropic openai feedparser pyyaml
   ```

2. **Set up API keys**:
   ```bash
   export GEMINI_API_KEY="your_key_here"
   ```

3. **Copy example config**:
   ```bash
   cp poly_utils/config.example.yaml my_bot/config.yaml
   ```

4. **Import and use**:
   ```python
   from poly_utils import LLMClient, NewsFeed

   llm = LLMClient(config['llm'])
   feed = NewsFeed(config['news'])
   ```

---

## 📚 See Also

- **spike_momentum**: Full implementation using LLM + News for sports arbitrage
- **config.example.yaml**: Complete configuration reference
- **llm_client.py**: LLM client source code
- **news_feed.py**: News feed source code

---

## 🤝 Contributing

When adding new features to poly_utils:

1. Keep interfaces general and reusable
2. Add comprehensive docstrings
3. Update this README with examples
4. Update `config.example.yaml` with new options
5. Test with multiple bots to ensure compatibility

---

## 📝 License

MIT License - See main repo for details
