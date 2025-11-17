"""
LLM Provider Abstraction Layer

Unified interface for multiple LLM providers:
- Google Gemini (recommended for speed + cost)
- Anthropic Claude
- OpenAI GPT
- OpenRouter (multi-model access)

Usage:
    provider = LLMProvider(config)
    response = provider.analyze(prompt, json_mode=True)
"""

import os
import json
import time
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from poly_utils.logging_utils import get_logger

logger = get_logger('spike_momentum.llm')


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.timeout = config.get('timeout', 10)
        self.max_retries = config.get('max_retries', 2)

    @abstractmethod
    def call(self, prompt: str, json_mode: bool = False) -> Dict[str, Any]:
        """
        Make LLM API call.

        Args:
            prompt: Input prompt
            json_mode: If True, request JSON response

        Returns:
            Dict with 'content' and 'success' keys
        """
        pass


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config.get('gemini', {}).get('model', 'gemini-2.5-flash')
        api_key_env = config.get('gemini', {}).get('api_key_env', 'GEMINI_API_KEY')
        self.api_key = os.getenv(api_key_env)

        if not self.api_key:
            raise ValueError(f"Missing API key: {api_key_env} not set in environment")

        # Import Gemini SDK
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.genai = genai
            self.client = genai.GenerativeModel(self.model)
        except ImportError:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")

    def call(self, prompt: str, json_mode: bool = False) -> Dict[str, Any]:
        """Call Gemini API."""
        try:
            # Configure generation
            generation_config = {
                'temperature': 0.3,  # Lower for more consistent analysis
                'top_p': 0.95,
                'top_k': 40,
                'max_output_tokens': 1024,
            }

            if json_mode:
                generation_config['response_mime_type'] = 'application/json'

            # Make API call
            response = self.client.generate_content(
                prompt,
                generation_config=generation_config,
                request_options={'timeout': self.timeout}
            )

            content = response.text

            # Parse JSON if requested
            if json_mode:
                try:
                    content = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON response: {e}")
                    return {'success': False, 'error': 'Invalid JSON response'}

            return {
                'success': True,
                'content': content,
                'model': self.model,
                'provider': 'gemini'
            }

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return {'success': False, 'error': str(e)}


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config.get('anthropic', {}).get('model', 'claude-haiku-3-5')
        api_key_env = config.get('anthropic', {}).get('api_key_env', 'ANTHROPIC_API_KEY')
        self.api_key = os.getenv(api_key_env)

        if not self.api_key:
            raise ValueError(f"Missing API key: {api_key_env} not set in environment")

        # Import Anthropic SDK
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic not installed. Run: pip install anthropic")

    def call(self, prompt: str, json_mode: bool = False) -> Dict[str, Any]:
        """Call Claude API."""
        try:
            messages = [{"role": "user", "content": prompt}]

            # Add JSON instruction if requested
            if json_mode:
                prompt += "\n\nRespond with valid JSON only."

            # Make API call
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
                timeout=self.timeout
            )

            content = response.content[0].text

            # Parse JSON if requested
            if json_mode:
                try:
                    content = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON response: {e}")
                    return {'success': False, 'error': 'Invalid JSON response'}

            return {
                'success': True,
                'content': content,
                'model': self.model,
                'provider': 'anthropic'
            }

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return {'success': False, 'error': str(e)}


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config.get('openai', {}).get('model', 'gpt-4o-mini')
        api_key_env = config.get('openai', {}).get('api_key_env', 'OPENAI_API_KEY')
        self.api_key = os.getenv(api_key_env)

        if not self.api_key:
            raise ValueError(f"Missing API key: {api_key_env} not set in environment")

        # Import OpenAI SDK
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, timeout=self.timeout)
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")

    def call(self, prompt: str, json_mode: bool = False) -> Dict[str, Any]:
        """Call OpenAI API."""
        try:
            # Configure request
            kwargs = {
                'model': self.model,
                'messages': [{"role": "user", "content": prompt}],
                'temperature': 0.3,
                'max_tokens': 1024,
            }

            if json_mode:
                kwargs['response_format'] = {"type": "json_object"}
                prompt += "\n\nRespond with valid JSON only."
                kwargs['messages'] = [{"role": "user", "content": prompt}]

            # Make API call
            response = self.client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content

            # Parse JSON if requested
            if json_mode:
                try:
                    content = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON response: {e}")
                    return {'success': False, 'error': 'Invalid JSON response'}

            return {
                'success': True,
                'content': content,
                'model': self.model,
                'provider': 'openai'
            }

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return {'success': False, 'error': str(e)}


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter provider (multi-model access via OpenAI-compatible API)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config.get('openrouter', {}).get('model', 'google/gemini-2.5-flash')
        api_key_env = config.get('openrouter', {}).get('api_key_env', 'OPENROUTER_API_KEY')
        self.api_key = os.getenv(api_key_env)

        if not self.api_key:
            raise ValueError(f"Missing API key: {api_key_env} not set in environment")

        # Import OpenAI SDK (OpenRouter is compatible)
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1",
                timeout=self.timeout
            )
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")

    def call(self, prompt: str, json_mode: bool = False) -> Dict[str, Any]:
        """Call OpenRouter API."""
        try:
            # Configure request
            kwargs = {
                'model': self.model,
                'messages': [{"role": "user", "content": prompt}],
                'temperature': 0.3,
                'max_tokens': 1024,
            }

            if json_mode:
                prompt += "\n\nRespond with valid JSON only."
                kwargs['messages'] = [{"role": "user", "content": prompt}]

            # Make API call
            response = self.client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content

            # Parse JSON if requested
            if json_mode:
                try:
                    content = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON response: {e}")
                    return {'success': False, 'error': 'Invalid JSON response'}

            return {
                'success': True,
                'content': content,
                'model': self.model,
                'provider': 'openrouter'
            }

        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            return {'success': False, 'error': str(e)}


class LLMProvider:
    """
    Unified LLM provider interface.

    Automatically selects the appropriate provider based on config.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider_name = config.get('provider', 'gemini').lower()

        # Initialize appropriate provider
        if self.provider_name == 'gemini':
            self.provider = GeminiProvider(config)
        elif self.provider_name == 'anthropic':
            self.provider = AnthropicProvider(config)
        elif self.provider_name == 'openai':
            self.provider = OpenAIProvider(config)
        elif self.provider_name == 'openrouter':
            self.provider = OpenRouterProvider(config)
        else:
            raise ValueError(f"Unknown provider: {self.provider_name}")

        logger.info(f"Initialized LLM provider: {self.provider_name}")

        # Cost tracking
        self.total_calls = 0
        self.total_cost = 0.0
        self.hourly_calls = {}  # timestamp -> count

    def analyze(self, prompt: str, json_mode: bool = True, retries: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze prompt with LLM.

        Args:
            prompt: Input prompt
            json_mode: If True, request JSON response
            retries: Number of retries (defaults to config value)

        Returns:
            Dict with analysis results
        """
        if retries is None:
            retries = self.provider.max_retries

        # Check rate limits
        if not self._check_rate_limits():
            logger.warning("Rate limit exceeded, skipping LLM call")
            return {'success': False, 'error': 'Rate limit exceeded'}

        # Attempt API call with retries
        for attempt in range(retries + 1):
            response = self.provider.call(prompt, json_mode=json_mode)

            if response['success']:
                self._track_call()
                return response

            if attempt < retries:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Retry {attempt + 1}/{retries} after {wait_time}s...")
                time.sleep(wait_time)

        # All retries failed
        logger.error("All LLM API retries failed")
        return {'success': False, 'error': 'Max retries exceeded'}

    def _check_rate_limits(self) -> bool:
        """Check if within rate limits."""
        max_calls_per_hour = self.config.get('max_calls_per_hour', 100)
        max_cost_per_day = self.config.get('max_cost_per_day', 10.0)

        # Check hourly limit
        current_hour = int(time.time() // 3600)
        hourly_count = sum(count for hour, count in self.hourly_calls.items() if hour == current_hour)

        if hourly_count >= max_calls_per_hour:
            return False

        # Check daily cost (simplified - actual cost calculation would need token counting)
        if self.total_cost >= max_cost_per_day:
            return False

        return True

    def _track_call(self):
        """Track API call for rate limiting and cost."""
        self.total_calls += 1
        current_hour = int(time.time() // 3600)

        if current_hour not in self.hourly_calls:
            self.hourly_calls[current_hour] = 0

        self.hourly_calls[current_hour] += 1

        # Simplified cost estimation (would need actual token counts)
        estimated_cost = 0.001  # rough estimate per call
        self.total_cost += estimated_cost

        # Cleanup old hourly data
        cutoff = current_hour - 24
        self.hourly_calls = {h: c for h, c in self.hourly_calls.items() if h > cutoff}

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            'provider': self.provider_name,
            'total_calls': self.total_calls,
            'estimated_cost': self.total_cost,
            'hourly_calls': sum(self.hourly_calls.values())
        }
