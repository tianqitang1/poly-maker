"""
LLM Strategy Module for LLM Maker Bot

Handles LLM integration:
- Generates prompts for market analysis
- Parses LLM responses into trading signals
- Maintains decision history
"""

import sys
import os
import json
import time
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from poly_utils.llm_client import LLMClient
from poly_utils.logging_utils import get_logger

logger = get_logger('llm_maker.strategy')


class LLMStrategy:
    """Manages LLM-based trading strategy decisions."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM strategy.

        Args:
            config: Full configuration dict with 'llm' and 'strategy' sections
        """
        self.config = config
        self.llm_config = config.get('llm', {})
        self.strategy_config = config.get('strategy', {})

        # Initialize LLM client
        if self.llm_config.get('enabled', True):
            self.llm_client = LLMClient(self.llm_config, bot_name='llm_maker')
            logger.info("LLM client initialized successfully")
        else:
            self.llm_client = None
            logger.warning("LLM client disabled in config")

        # Track failures for fallback
        self.consecutive_failures = 0
        self.fallback_mode = False
        self.last_decision_time = 0

        # Decision history (for logging and analysis)
        self.decision_history = []

    def generate_trading_signals(self, llm_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to generate trading signals for markets.

        Args:
            llm_input: Market data prepared by MarketAnalyzer

        Returns:
            Dict with trading signals:
            {
                'success': bool,
                'signals': [{'market_id': str, 'action': str, 'confidence': float, ...}],
                'timestamp': str,
                'model_used': str
            }
        """
        # Check if we should use fallback
        fallback_threshold = self.strategy_config.get('fallback_after_failures', 3)
        if self.consecutive_failures >= fallback_threshold:
            if not self.fallback_mode:
                logger.warning(f"Switching to fallback mode after {self.consecutive_failures} consecutive failures")
                self.fallback_mode = True
            return self._generate_fallback_signals(llm_input)

        # Check if LLM is enabled
        if not self.llm_client:
            return self._generate_fallback_signals(llm_input)

        # Generate prompt
        prompt = self._generate_prompt(llm_input)

        # Query LLM
        try:
            response = self.llm_client.query(prompt, json_mode=True, temperature=0.3, max_tokens=2048)

            if not response['success']:
                logger.error(f"LLM query failed: {response.get('error', 'Unknown error')}")
                self.consecutive_failures += 1
                return self._generate_fallback_signals(llm_input)

            # Parse response
            signals = self._parse_llm_response(response['content'], llm_input)

            # Reset failure count on success
            self.consecutive_failures = 0
            self.fallback_mode = False

            # Log decision
            self._log_decision(llm_input, signals, response)

            return {
                'success': True,
                'signals': signals,
                'timestamp': llm_input['timestamp'],
                'model_used': response.get('model', 'unknown'),
                'provider': response.get('provider', 'unknown')
            }

        except Exception as e:
            logger.error(f"Error in LLM strategy: {e}", exc_info=True)
            self.consecutive_failures += 1
            return self._generate_fallback_signals(llm_input)

    def _generate_prompt(self, llm_input: Dict[str, Any]) -> str:
        """
        Generate prompt for LLM.

        Args:
            llm_input: Market data from MarketAnalyzer

        Returns:
            Formatted prompt string
        """
        portfolio = llm_input['portfolio']
        markets = llm_input['markets']

        prompt = f"""You are a market making AI for prediction markets on Polymarket. Your job is to analyze markets and provide trading signals.

**Current Portfolio:**
- Active Positions: {portfolio['num_positions']}
- Total Exposure: ${portfolio['total_exposure']}
- Available Capital: ${portfolio['available_capital']}
- Capital Utilization: {portfolio['capital_utilization']*100:.1f}%

**Markets to Analyze:**
"""

        for i, market in enumerate(markets, 1):
            prompt += f"""
{i}. **{market['question']}**
   Answers: {market['answer1']} vs {market['answer2']}

   {market['answer1']} (YES):
   - Price: {market['token1']['mid_price']} | Spread: {market['token1']['spread']}
   - Orderbook Imbalance: {market['token1']['orderbook_imbalance']:.2f} (>1 = more buyers)
   - Trend: {market['token1']['trend']}
   - Our Position: {market['token1']['position']['size']} shares @ {market['token1']['position']['avg_price']} (P&L: {market['token1']['position']['pnl_percent']}%)

   {market['answer2']} (NO):
   - Price: {market['token2']['mid_price']} | Spread: {market['token2']['spread']}
   - Orderbook Imbalance: {market['token2']['orderbook_imbalance']:.2f}
   - Trend: {market['token2']['trend']}
   - Our Position: {market['token2']['position']['size']} shares @ {market['token2']['position']['avg_price']} (P&L: {market['token2']['position']['pnl_percent']}%)

   Metadata: Volatility: {market['metadata']['volatility_3h']:.2%}, Trade Size: {market['metadata']['trade_size']}, Max Size: {market['metadata']['max_size']}
"""

        prompt += """

**Your Task:**
For each market, provide a trading signal with the following:
1. **action**: "buy_yes", "buy_no", "sell_yes", "sell_no", "hold", "exit"
2. **confidence**: 0.0-1.0 (how confident are you in this signal)
3. **reasoning**: Brief explanation (1-2 sentences)
4. **directional_bias**: "bullish", "bearish", "neutral" (for that outcome)
5. **risk_level**: "low", "medium", "high"
6. **priority**: "high", "medium", "low" (which markets to focus on)

**Guidelines:**
- **exit**: Recommend if P&L < -5% (stop-loss) or > 8% (take-profit) or if market conditions deteriorated
- **hold**: If we have a position and market conditions are stable
- **buy**: If spread is tight, liquidity is good, and direction is favorable
- **sell**: To exit or reduce position
- Consider orderbook imbalance (>1.3 = strong buying pressure, <0.7 = strong selling pressure)
- Consider trends (buy into "up" trends carefully, sell into "down" trends quickly)
- Prefer markets with tight spreads (<0.05) and stable volatility
- Exit positions early if volatility > 10%

**Response Format (JSON only):**
```json
{
  "signals": [
    {
      "market_id": "condition_id_here",
      "action": "buy_yes|buy_no|sell_yes|sell_no|hold|exit",
      "confidence": 0.75,
      "reasoning": "Your brief explanation",
      "directional_bias": "bullish|bearish|neutral",
      "risk_level": "low|medium|high",
      "priority": "high|medium|low"
    }
  ],
  "portfolio_assessment": {
    "overall_risk": "low|medium|high",
    "suggested_actions": "Brief summary of recommended actions"
  }
}
```

Respond with ONLY the JSON above, no additional text.
"""

        return prompt

    def _parse_llm_response(self, response_content: Any, llm_input: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse LLM JSON response into trading signals.

        Args:
            response_content: Parsed JSON response from LLM
            llm_input: Original market data (for validation)

        Returns:
            List of trading signal dicts
        """
        if not isinstance(response_content, dict):
            logger.warning("LLM response is not a dict, attempting to parse as JSON")
            try:
                response_content = json.loads(response_content)
            except:
                logger.error("Failed to parse LLM response as JSON")
                return []

        signals = response_content.get('signals', [])

        # Validate and clean up signals
        valid_signals = []
        for signal in signals:
            # Validate required fields
            if not all(key in signal for key in ['market_id', 'action', 'confidence']):
                logger.warning(f"Skipping invalid signal (missing required fields): {signal}")
                continue

            # Validate confidence
            try:
                confidence = float(signal['confidence'])
                if not (0.0 <= confidence <= 1.0):
                    logger.warning(f"Invalid confidence {confidence}, clamping to [0, 1]")
                    confidence = max(0.0, min(1.0, confidence))
                signal['confidence'] = confidence
            except (ValueError, TypeError):
                logger.warning(f"Invalid confidence value: {signal.get('confidence')}")
                continue

            # Validate action
            valid_actions = ['buy_yes', 'buy_no', 'sell_yes', 'sell_no', 'hold', 'exit']
            if signal['action'] not in valid_actions:
                logger.warning(f"Invalid action: {signal['action']}")
                continue

            # Add defaults for optional fields
            signal.setdefault('reasoning', 'No reasoning provided')
            signal.setdefault('directional_bias', 'neutral')
            signal.setdefault('risk_level', 'medium')
            signal.setdefault('priority', 'medium')

            valid_signals.append(signal)

        logger.info(f"Parsed {len(valid_signals)} valid signals from LLM response")
        return valid_signals

    def _generate_fallback_signals(self, llm_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals using simple pre-configured logic.

        This is used when LLM is unavailable or failing.

        Args:
            llm_input: Market data from MarketAnalyzer

        Returns:
            Dict with fallback signals
        """
        logger.info("Generating fallback signals (LLM unavailable)")

        signals = []
        for market in llm_input['markets']:
            # Simple rule-based signals
            signal = {
                'market_id': market['condition_id'],
                'action': 'hold',  # Default to hold
                'confidence': 0.5,
                'reasoning': 'Fallback mode - simple rules',
                'directional_bias': 'neutral',
                'risk_level': 'low',
                'priority': 'low'
            }

            # Check for stop-loss conditions
            for side in ['token1', 'token2']:
                pos = market[side]['position']
                if pos['pnl_percent'] and pos['pnl_percent'] < -5:
                    signal['action'] = 'exit'
                    signal['reasoning'] = f"Stop-loss triggered: {pos['pnl_percent']:.1f}% loss"
                    signal['priority'] = 'high'
                    break

                # Check for take-profit
                if pos['pnl_percent'] and pos['pnl_percent'] > 8:
                    signal['action'] = 'exit'
                    signal['reasoning'] = f"Take-profit triggered: {pos['pnl_percent']:.1f}% gain"
                    signal['priority'] = 'high'
                    break

            signals.append(signal)

        return {
            'success': True,
            'signals': signals,
            'timestamp': llm_input['timestamp'],
            'model_used': 'fallback',
            'provider': 'rule-based'
        }

    def _log_decision(self, llm_input: Dict[str, Any], signals: List[Dict[str, Any]], llm_response: Dict[str, Any]):
        """
        Log LLM decision for analysis.

        Args:
            llm_input: Original market data
            signals: Parsed trading signals
            llm_response: Raw LLM response
        """
        decision_record = {
            'timestamp': llm_input['timestamp'],
            'model': llm_response.get('model', 'unknown'),
            'provider': llm_response.get('provider', 'unknown'),
            'num_signals': len(signals),
            'signals': signals,
            'portfolio_state': llm_input['portfolio']
        }

        # Add to history
        self.decision_history.append(decision_record)

        # Keep only last 100 decisions in memory
        if len(self.decision_history) > 100:
            self.decision_history = self.decision_history[-100:]

        # Write to log file if configured
        log_file = self.config.get('logging', {}).get('decision_log_file')
        if log_file and self.config.get('logging', {}).get('log_llm_decisions', True):
            try:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                with open(log_file, 'a') as f:
                    f.write(json.dumps(decision_record) + '\n')
            except Exception as e:
                logger.error(f"Failed to write decision log: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get LLM strategy statistics.

        Returns:
            Dict with strategy stats
        """
        stats = {
            'consecutive_failures': self.consecutive_failures,
            'fallback_mode': self.fallback_mode,
            'decisions_made': len(self.decision_history),
            'last_decision_time': self.last_decision_time
        }

        if self.llm_client:
            stats['llm_client'] = self.llm_client.get_stats()

        return stats
