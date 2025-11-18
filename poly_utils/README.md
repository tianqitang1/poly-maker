# Poly Utils - Shared Utilities for Poly-Maker Bots

This package provides generalized, reusable utilities for all trading bots in the poly-maker suite.

## Features

- **LLM Client**: Unified interface for multiple LLM providers (Gemini, Claude, GPT, etc.)
- **News Feed**: Multi-category news aggregator (sports, crypto, politics, general)
- **Semantic Search**: ChromaDB-based semantic matching (much better than keyword matching!)
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

## 📡 Next News API Integration (Optional)

Integrate with [next-news-api](https://github.com/riad-azz/next-news-api) for access to multiple aggregated news sources (Google News, NewsAPI, NewsData) via a unified API. The server automatically starts/stops as needed.

### Why Next News API?

- **Multiple sources in one**: Google News, NewsAPI, NewsData, and more
- **Auto-managed**: Server starts automatically when needed
- **No manual setup**: Dependencies and build handled automatically
- **Free sources available**: Google News requires no API key
- **Unified interface**: Same API for all sources

### Setup

1. **Clone next-news-api**:
   ```bash
   git clone https://github.com/riad-azz/next-news-api ~/next-news-api
   ```

2. **Configure in poly_utils**:
   ```yaml
   news:
     sources:
       next_news_api:
         enabled: true
         server_path: "~/next-news-api"
         port: 3000
         auto_start: true
         sources:
           - google  # Free, no API key needed
         max_results_per_source: 10
   ```

3. **Use automatically**:
   ```python
   from poly_utils import NewsFeed

   feed = NewsFeed(config)
   # Server auto-starts on first request
   news = feed.fetch_news(category='sports')
   # Server auto-stops on exit
   ```

### Direct Usage

You can also use NextNewsAPIManager directly:

```python
from poly_utils import NextNewsAPIManager

config = {
    'enabled': True,
    'server_path': '~/next-news-api',
    'port': 3000,
    'auto_start': True
}

manager = NextNewsAPIManager(config)

# Fetch from single source
news = manager.fetch_news(
    source='google',
    query='cryptocurrency',
    category='technology',
    max_results=10
)

# Fetch from multiple sources
news = manager.fetch_multiple_sources(
    sources=['google', 'newsapi'],
    query='NBA playoffs',
    category='sports',
    max_results_per_source=5
)

# Check status
status = manager.get_status()
print(f"Server running: {status['running']}")
print(f"Available sources: {status['available_sources']}")

# Manual cleanup (optional - auto-cleanup on exit)
manager.stop_server()
```

### Configuration Options

```yaml
next_news_api:
  enabled: true
  server_path: "~/next-news-api"    # Path to cloned repo
  port: 3000                          # Server port
  host: "localhost"                   # Server host
  auto_start: true                    # Auto-start if not running
  startup_timeout: 30                 # Seconds to wait for startup

  sources:                            # News sources to use
    - google                          # Google News (free)
    - newsapi                         # NewsAPI.org (requires API key)
    - newsdata                        # NewsData.io (requires API key)

  query: null                         # Optional search query filter
  max_results_per_source: 10          # Max results per source
```

### Available Sources

| Source | API Key Required | Cost | Coverage |
|--------|-----------------|------|----------|
| **google** | ❌ No | Free | Global news |
| newsapi | ✅ Yes | Free tier available | 80+ countries |
| newsdata | ✅ Yes | Free tier available | 48+ countries |

**Recommendation**: Start with Google News (free, no API key). Add other sources as needed.

### Features

**Automatic Server Management:**
- Server starts automatically on first request
- Health checks before each request
- Auto-restart if server crashes
- Clean shutdown on exit

**Error Handling:**
- Graceful fallback if server fails to start
- Timeout protection
- Automatic dependency installation
- Build on first run

**Integration with NewsFeed:**
- Seamlessly integrated as another news source
- Automatic category detection
- Deduplication with RSS sources
- Compatible with semantic search

### Example: Sports News with Next News API

```python
config = {
    'enabled': True,
    'categories': ['sports'],
    'sources': {
        'espn_rss': {'enabled': True, 'leagues': ['nfl', 'nba']},
        'next_news_api': {
            'enabled': True,
            'server_path': '~/next-news-api',
            'sources': ['google'],
            'max_results_per_source': 10
        }
    }
}

feed = NewsFeed(config)

# Fetches from both ESPN RSS and Google News (via next-news-api)
news = feed.fetch_news(category='sports', max_items=20)

# Automatic deduplication combines unique items from both sources
for item in news:
    print(f"[{item.source}] {item.title}")
```

---

## 🔍 Semantic Search (RECOMMENDED!)

Semantic search uses embeddings to find relevant news instead of keyword matching. **This dramatically improves matching accuracy.**

### Why Semantic Search?

**Problem with keyword matching:**
```
Market: "Will Chet Holmgren win the 2025-2026 NBA Defensive Player of the Year?"

Keyword matches:
  ❌ "Players react to NFL's efforts to halt team report cards" (0.29)
  ❌ "Kempe, Kings reach 8-year, $85M deal" (0.29)
  ❌ "Grading bold season predictions for all 30 MLB teams" (0.09)

All completely irrelevant!
```

**With semantic search:**
```
Market: "Will Chet Holmgren win the 2025-2026 NBA Defensive Player of the Year?"

Semantic matches:
  ✅ "Holmgren records 5 blocks in Thunder win over Mavs" (0.87)
  ✅ "Thunder's defense dominates as Holmgren anchors paint" (0.82)
  ✅ "OKC climbs to #1 defensive rating with Holmgren leading" (0.79)

Highly relevant and actually useful!
```

### Installation

```bash
# Required dependencies
pip install chromadb sentence-transformers

# Optional cloud providers
pip install openai cohere  # For OpenAI/Cohere embeddings
```

### Basic Usage

```python
from poly_utils import NewsFeed
import yaml

# Load config with semantic search enabled
config = {
    'enabled': True,
    'categories': ['sports'],
    'sources': {
        'espn_rss': {'enabled': True, 'leagues': ['nfl', 'nba']}
    },
    'semantic_search': {
        'enabled': True,
        'provider': 'sentence_transformer',  # Local, free
        'model': 'all-MiniLM-L6-v2',
        'similarity_threshold': 0.5
    }
}

feed = NewsFeed(config)

# Fetch news (automatically indexed for semantic search)
news = feed.fetch_news(category='sports')

# Semantic matching (much better than keyword matching!)
matches = feed.semantic_match_to_market(
    market_question="Will Chet Holmgren win NBA DPOY?",
    market_id="holmgren-dpoy-2026",  # Optional: enables caching
    max_results=5,
    min_similarity=0.6  # Override threshold
)

for match in matches:
    news_item = match['news']
    score = match['relevance_score']
    print(f"[{score:.2f}] {news_item.title}")
```

### Embedding Providers

| Provider | Model | Location | Cost | Quality | Speed |
|----------|-------|----------|------|---------|-------|
| **sentence_transformer** | all-MiniLM-L6-v2 | Local | Free | Good | Very Fast |
| sentence_transformer | all-mpnet-base-v2 | Local | Free | Better | Fast |
| **openai** | text-embedding-3-small | Cloud | $0.02/1M tokens | Excellent | Fast |
| openai | text-embedding-3-large | Cloud | $0.13/1M tokens | Best | Fast |
| cohere | embed-english-v3.0 | Cloud | $0.10/1M tokens | Excellent | Fast |
| gemini | embedding-001 | Cloud | ~$0.00001/1K chars | Good | Fast |

**Recommendation**: Use `sentence_transformer` (local, free) for most cases. Use cloud providers for higher accuracy on complex queries.

### Market Embedding Cache

Semantic search automatically caches market embeddings to avoid re-computing them:

```python
# Cache market embedding for faster repeat searches
feed.semantic_match_to_market(
    market_question="Will Bitcoin reach $100k?",
    market_id="btc-100k",  # Cached for future queries
)

# Second query uses cached embedding (instant!)
feed.semantic_match_to_market(
    market_question="Will Bitcoin reach $100k?",
    market_id="btc-100k",  # Retrieved from cache
)
```

### Cleanup

Automatically remove closed/old markets from cache:

```python
# Remove specific closed markets
closed_ids = ["market-123", "market-456"]
feed.cleanup_closed_markets(closed_ids)

# Remove markets older than 7 days
feed.cleanup_old_markets(max_age_hours=168)
```

### Configuration

```yaml
news:
  semantic_search:
    enabled: true
    provider: "sentence_transformer"  # or openai, cohere, gemini
    model: "all-MiniLM-L6-v2"
    similarity_threshold: 0.5         # 0-1 (higher = more strict)
    cache_markets: true                # Enable market caching
    chroma_dir: ".cache/chromadb"      # Storage location
```

### Advanced Usage with Direct Access

```python
from poly_utils import SemanticSearchEngine

# Direct access to semantic search engine
config = {
    'enabled': True,
    'provider': 'sentence_transformer',
    'model': 'all-MiniLM-L6-v2'
}

engine = SemanticSearchEngine(config)

# Add news manually
engine.add_news(news_items)

# Search with custom parameters
results = engine.search_news(
    query="Will Lakers win tonight?",
    market_id="lakers-game-123",
    max_results=10,
    min_similarity=0.7
)

# Get stats
stats = engine.get_stats()
print(f"Total embeddings: {stats['embedding_stats']['total_embeddings']}")
print(f"Cached markets: {stats['cached_markets']}")
```

### Cost Comparison

**Semantic search costs (per 1,000 news items):**

| Provider | Cost | Notes |
|----------|------|-------|
| sentence_transformer | **$0** | Local, free, one-time download |
| gemini | ~$0.01 | Very cheap |
| openai (small) | ~$0.02 | Good quality/price ratio |
| cohere | ~$0.10 | Optimized for search |
| openai (large) | ~$0.13 | Highest quality |

**Recommendation**: Start with local `sentence_transformer` (free). Upgrade to cloud only if accuracy isn't sufficient.

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
   from poly_utils import LLMClient, NewsFeed, SemanticSearchEngine

   llm = LLMClient(config['llm'])
   feed = NewsFeed(config['news'])

   # Semantic search is automatically enabled if configured in config['news']['semantic_search']
   # Or use directly:
   # semantic_engine = SemanticSearchEngine(config['semantic_search'])
   ```

---

## 📚 See Also

- **spike_momentum**: Full implementation using LLM + News + Semantic Search for sports arbitrage
- **config.example.yaml**: Complete configuration reference including semantic search
- **llm_client.py**: LLM client source code
- **news_feed.py**: News feed source code
- **semantic_search.py**: Semantic search engine source code

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
