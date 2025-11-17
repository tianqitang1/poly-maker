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
        news_items: List[NewsItem],
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Optional[SpikeAnalysis]:
        """
        Analyze whether a price spike is justified.

        Args:
            market_question: The Polymarket question
            current_price: Current market price (0-1)
            previous_price: Price before spike (0-1)
            price_change_pct: Percentage change (e.g., 0.15 for 15%)
            news_items: Relevant news items
            additional_context: Optional extra context (game time, score, etc.)

        Returns:
            SpikeAnalysis object or None if analysis failed
        """
        # Build prompt
        prompt = self._build_sports_prompt(
            market_question=market_question,
            current_price=current_price,
            previous_price=previous_price,
            price_change_pct=price_change_pct,
            news_items=news_items,
            additional_context=additional_context
        )

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
        news_items: List[NewsItem],
        additional_context: Optional[Dict[str, Any]]
    ) -> str:
        """Build LLM prompt for sports market analysis."""

        # Format news items
        news_text = self._format_news_items(news_items)

        # Build prompt
        prompt = f"""You are an expert sports analyst evaluating prediction market price movements.

MARKET QUESTION: "{market_question}"

PRICE MOVEMENT:
- Previous price: {previous_price:.3f} ({previous_price*100:.1f}% probability)
- Current price: {current_price:.3f} ({current_price*100:.1f}% probability)
- Change: {price_change_pct:+.1f}% ({"up" if price_change_pct > 0 else "down"})

RECENT NEWS:
{news_text}

"""

        # Add additional context if provided
        if additional_context:
            context_text = self._format_additional_context(additional_context)
            if context_text:
                prompt += f"""ADDITIONAL CONTEXT:
{context_text}

"""

        # Add analysis instructions
        prompt += """ANALYSIS TASK:
Evaluate whether this price movement is justified based on the news and context.

Consider:
1. Does the news directly relate to this market question?
2. Does the news support the direction of the price movement?
3. Is this information likely already priced in, or is it fresh?
4. How close is this market to resolution? (game ending, event concluded, etc.)
5. What is the confidence level that this price movement is justified?

OUTPUT FORMAT (JSON):
{
  "justified": true or false,
  "confidence": 0-100 (how confident you are in your analysis),
  "reasoning": "detailed explanation of your analysis",
  "near_resolution": true or false (is this market close to resolving?),
  "estimated_time_to_resolution": "e.g., '5 minutes', '2 hours', 'tomorrow', null if unknown",
  "recommendation": "buy", "sell", or "hold",
  "risk_factors": ["list", "of", "potential", "risks"]
}

Think step-by-step about the probability of the outcome given the current situation.
For sports: consider game time remaining, current score, typical win probabilities.
Be objective and data-driven. Avoid being swayed by excitement or narratives.

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
