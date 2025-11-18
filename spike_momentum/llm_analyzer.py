"""
LLM Analyzer for Spike Momentum Trading

Uses LLM (Gemini/Claude/GPT) to analyze whether price spikes are justified
by underlying news/events, with special focus on sports markets.
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from spike_momentum.llm_provider import LLMProvider
from spike_momentum.news_monitor import NewsItem
from poly_utils.logging_utils import get_logger

logger = get_logger('spike_momentum.llm_analyzer')


class SpikeAnalysis:
    """Represents LLM analysis of a price spike."""

    def __init__(self, data: Dict[str, Any]):
        self.justified = data.get('justified', False)
        self.confidence = data.get('confidence', 0)
        self.reasoning = data.get('reasoning', '')
        self.near_resolution = data.get('near_resolution', False)
        self.estimated_time_to_resolution = data.get('estimated_time_to_resolution', None)
        self.recommendation = data.get('recommendation', 'hold')
        self.risk_factors = data.get('risk_factors', [])
        self.raw_response = data

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'justified': self.justified,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'near_resolution': self.near_resolution,
            'estimated_time_to_resolution': self.estimated_time_to_resolution,
            'recommendation': self.recommendation,
            'risk_factors': self.risk_factors
        }

    def __repr__(self) -> str:
        return f"<SpikeAnalysis: justified={self.justified}, confidence={self.confidence}%>"


class LLMAnalyzer:
    """Analyzes price spikes using LLM."""

    def __init__(self, llm_provider: LLMProvider, config: Dict[str, Any]):
        """
        Initialize LLM analyzer.

        Args:
            llm_provider: Initialized LLM provider
            config: LLM configuration from config.yaml
        """
        self.llm = llm_provider
        self.config = config
        self.min_confidence = config.get('min_confidence', 70)

        logger.info(f"Initialized LLMAnalyzer (min_confidence={self.min_confidence})")

    def analyze_spike(
        self,
        market_question: str,
        current_price: float,
        previous_price: float,
        price_change_pct: float,
        game_metadata: Optional[Dict[str, Any]] = None,
        market_description: Optional[str] = None,
        news_items: Optional[List[NewsItem]] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Optional[SpikeAnalysis]:
        """
        Analyze whether a price spike is justified using Polymarket game data.

        Args:
            market_question: The Polymarket question
            current_price: Current market price (0-1)
            previous_price: Price before spike (0-1)
            price_change_pct: Percentage change (e.g., 0.15 for 15%)
            game_metadata: Real-time game data from Polymarket (score, period, live, etc.)
            market_description: Market description text
            news_items: Optional news items (fallback if game_metadata unavailable)
            additional_context: Optional extra context

        Returns:
            SpikeAnalysis object or None if analysis failed
        """
        # Build prompt using game metadata (preferred) or news (fallback)
        prompt = self._build_sports_prompt(
            market_question=market_question,
            current_price=current_price,
            previous_price=previous_price,
            price_change_pct=price_change_pct,
            game_metadata=game_metadata,
            market_description=market_description,
            news_items=news_items or [],
            additional_context=additional_context
        )

        # Log the prompt being sent to LLM
        logger.debug(f"LLM Prompt for '{market_question[:50]}...':")
        logger.debug(f"Prompt:\n{prompt[:1000]}..." if len(prompt) > 1000 else f"Prompt:\n{prompt}")

        # Call LLM
        response = self.llm.analyze(prompt, json_mode=True)

        if not response['success']:
            logger.error(f"LLM analysis failed: {response.get('error')}")
            return None

        # Parse response
        try:
            data = response['content']
            analysis = SpikeAnalysis(data)

            logger.info(
                f"LLM Analysis: justified={analysis.justified}, "
                f"confidence={analysis.confidence}%, "
                f"recommendation={analysis.recommendation}"
            )

            return analysis

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return None

    def _build_sports_prompt(
        self,
        market_question: str,
        current_price: float,
        previous_price: float,
        price_change_pct: float,
        game_metadata: Optional[Dict[str, Any]] = None,
        market_description: Optional[str] = None,
        news_items: Optional[List[NewsItem]] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build LLM prompt for sports market analysis using Polymarket game data."""

        # Build market description section
        description_text = ""
        if market_description:
            description_text = f"""
MARKET DESCRIPTION:
{market_description}
"""

        # Build game data section (PRIORITY DATA!)
        game_data_text = ""
        if game_metadata:
            game_data_text = f"""
REAL-TIME GAME DATA (from Polymarket API):
- Live: {game_metadata.get('live', 'unknown')}
- Ended: {game_metadata.get('ended', 'unknown')}
- Score: {game_metadata.get('score', 'N/A')}
- Period: {game_metadata.get('period', 'N/A')}
- Elapsed Time: {game_metadata.get('elapsed', 'N/A')}
- Finished Timestamp: {game_metadata.get('finishedTimestamp', 'N/A')}
- Start Time: {game_metadata.get('startTime', 'N/A')}
"""

        # Format news items (OPTIONAL - only if game data missing)
        news_text = ""
        if news_items:
            formatted_news = self._format_news_items(news_items)
            news_text = f"""
RECENT NEWS (supplementary):
{formatted_news}
"""

        # Build prompt
        prompt = f"""You are an expert sports analyst evaluating prediction market price movements.

MARKET QUESTION: "{market_question}"
{description_text}
PRICE MOVEMENT:
- Previous price: {previous_price:.3f} ({previous_price*100:.1f}% probability)
- Current price: {current_price:.3f} ({current_price*100:.1f}% probability)
- Change: {price_change_pct:+.1f}% ({"up" if price_change_pct > 0 else "down"})
{game_data_text}{news_text}"""

        # Add additional context if provided
        if additional_context:
            context_text = self._format_additional_context(additional_context)
            if context_text:
                prompt += f"""
ADDITIONAL CONTEXT:
{context_text}
"""

        # Add analysis instructions
        prompt += """
ANALYSIS TASK:
Evaluate whether this price movement is justified based on the REAL-TIME GAME DATA and market context.

KEY CONSIDERATIONS:
1. GAME STATUS: Is the game live, ended, or not started?
2. SCORE IMPACT: If live, is the current score consistent with the price movement?
   - Examples: Team up 20 points → higher YES probability justified
   - Close game, late period → higher volatility expected
   - Blowout game → less uncertainty, price should be extreme
3. TIME REMAINING: How much game time is left? (from period + elapsed)
   - Late game (Q4, 9th inning) → high confidence in current leader
   - Early game (Q1, 1st inning) → lots of uncertainty, momentum can shift
4. PRICE SIGNAL: Does the price change match the game situation?
   - Trailing team price drops = justified
   - Winning team price rises = justified
   - Price spike without game change = potentially unjustified
5. RESOLUTION TIMING: How close is this market to resolving?

OUTPUT FORMAT (JSON):
{{
  "justified": true or false,
  "confidence": 0-100 (how confident you are in your analysis),
  "reasoning": "detailed explanation using game data (score, period, time, etc.)",
  "near_resolution": true or false (is this market close to resolving?),
  "estimated_time_to_resolution": "e.g., '2 minutes' (if Q4, 2:00 left), '1 hour' (if halftime), null if unknown",
  "recommendation": "buy" (price will continue up), "sell" (price will reverse), or "hold" (wait for more data),
  "risk_factors": ["list", "of", "potential", "risks", "based", "on", "game", "context"]
}}

SPORT-SPECIFIC ANALYSIS:
- Basketball: 10+ point lead in Q4 with <5 min = very high confidence for leader
- Football: 14+ point lead in Q4 with <5 min = high confidence (2+ possession game)
- Baseball: 3+ run lead in 9th = high confidence
- Soccer: 2+ goal lead in 80+ minute = high confidence
- Close games (<1 possession/goal) = medium confidence until final whistle

CRITICAL: Base your analysis on GAME DATA (score, period, time), not news headlines.
Think probabilistically: given the current score and time, what's the win probability?

Respond with valid JSON only."""

        return prompt

    def _format_news_items(self, news_items: List[NewsItem]) -> str:
        """Format news items for prompt."""
        if not news_items:
            return "(No recent news found)"

        formatted = []
        for i, item in enumerate(news_items[:5], 1):  # Top 5 items
            age_minutes = (datetime.now() - item.published).total_seconds() / 60
            formatted.append(
                f"{i}. [{item.source} - {age_minutes:.0f}m ago] {item.title}\n"
                f"   {item.summary[:200]}..."
            )

        return "\n".join(formatted)

    def _format_additional_context(self, context: Dict[str, Any]) -> str:
        """Format additional context for prompt."""
        lines = []

        # Game-specific context
        if 'game_time_remaining' in context:
            lines.append(f"- Game time remaining: {context['game_time_remaining']}")

        if 'score' in context:
            lines.append(f"- Current score: {context['score']}")

        if 'quarter' in context or 'period' in context:
            period = context.get('quarter') or context.get('period')
            lines.append(f"- Period: {period}")

        # General context
        if 'market_volume_24h' in context:
            lines.append(f"- 24h trading volume: ${context['market_volume_24h']:,.0f}")

        if 'market_liquidity' in context:
            lines.append(f"- Current liquidity: ${context['market_liquidity']:,.0f}")

        return "\n".join(lines) if lines else ""

    def should_trade(self, analysis: SpikeAnalysis) -> bool:
        """
        Determine if we should trade based on analysis.

        Args:
            analysis: SpikeAnalysis object

        Returns:
            True if confidence meets threshold and justified
        """
        if not analysis:
            return False

        # Must be justified
        if not analysis.justified:
            return False

        # Must meet confidence threshold
        if analysis.confidence < self.min_confidence:
            logger.info(
                f"Confidence {analysis.confidence}% below threshold {self.min_confidence}%"
            )
            return False

        # Must have a buy recommendation
        if analysis.recommendation not in ['buy', 'long']:
            return False

        return True

    def get_position_size_multiplier(self, analysis: SpikeAnalysis) -> float:
        """
        Calculate position size multiplier based on confidence.

        Args:
            analysis: SpikeAnalysis object

        Returns:
            Multiplier (0.5 - 2.0) based on confidence
        """
        if not analysis:
            return 1.0

        # Scale from 0.5x to 2.0x based on confidence
        # confidence 70 -> 0.5x
        # confidence 85 -> 1.0x
        # confidence 100 -> 2.0x

        confidence = analysis.confidence
        confidence = max(70, min(100, confidence))  # Clamp to 70-100

        # Linear scaling
        multiplier = 0.5 + (confidence - 70) * (1.5 / 30)

        return multiplier


def test_llm_analyzer():
    """Test the LLM analyzer."""
    import yaml
    from spike_momentum.llm_provider import LLMProvider
    from spike_momentum.news_monitor import NewsItem
    from datetime import datetime

    # Load config
    with open('spike_momentum/config.yaml.example') as f:
        config = yaml.safe_load(f)

    # Initialize LLM provider
    llm_provider = LLMProvider(config['llm'])

    # Initialize analyzer
    analyzer = LLMAnalyzer(llm_provider, config['llm'])

    # Create mock news
    news_items = [
        NewsItem(
            title="Lakers lead Warriors 98-89 with 2:14 remaining in 4th quarter",
            summary="LeBron James scores 8 straight points to give Lakers commanding lead late",
            link="https://espn.com/example",
            published=datetime.now(),
            source="ESPN",
            sport="NBA"
        )
    ]

    # Analyze spike
    print("Analyzing spike...")
    analysis = analyzer.analyze_spike(
        market_question="Will the Lakers beat the Warriors tonight?",
        current_price=0.78,
        previous_price=0.65,
        price_change_pct=20.0,
        news_items=news_items,
        additional_context={
            'game_time_remaining': '2:14',
            'score': 'Lakers 98, Warriors 89',
            'quarter': '4th'
        }
    )

    if analysis:
        print(f"\nAnalysis Results:")
        print(f"  Justified: {analysis.justified}")
        print(f"  Confidence: {analysis.confidence}%")
        print(f"  Reasoning: {analysis.reasoning}")
        print(f"  Near Resolution: {analysis.near_resolution}")
        print(f"  Time to Resolution: {analysis.estimated_time_to_resolution}")
        print(f"  Recommendation: {analysis.recommendation}")
        print(f"  Risk Factors: {', '.join(analysis.risk_factors)}")
        print(f"\n  Should Trade: {analyzer.should_trade(analysis)}")
        print(f"  Position Size Multiplier: {analyzer.get_position_size_multiplier(analysis):.2f}x")
    else:
        print("Analysis failed!")


if __name__ == '__main__':
    test_llm_analyzer()
