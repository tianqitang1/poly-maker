"""
LLM Provider Abstraction Layer

Unified interface for multiple LLM providers:
- Google Gemini (recommended for speed + cost)
- DeepSeek (very competitive pricing)
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
import re
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from collections import deque

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
                'max_completion_tokens': 1024,  # Updated from max_tokens for new API
            }

            # Some models don't support custom temperature
            # Try to detect based on model name
            models_without_temp = ['nano', 'o1', 'o3', 'mini']
            skip_temperature = any(name in self.model.lower() for name in models_without_temp)

            if not skip_temperature:
                kwargs['temperature'] = 0.3

            if json_mode:
                kwargs['response_format'] = {"type": "json_object"}
                prompt += "\n\nRespond with valid JSON only."
                kwargs['messages'] = [{"role": "user", "content": prompt}]

            # Make API call
            try:
                response = self.client.chat.completions.create(**kwargs)
            except Exception as api_error:
                # If temperature not supported, retry without it
                error_str = str(api_error)
                if 'temperature' in error_str.lower() and 'unsupported' in error_str.lower():
                    logger.warning(f"Model {self.model} doesn't support temperature parameter, retrying without it")
                    if 'temperature' in kwargs:
                        del kwargs['temperature']
                    response = self.client.chat.completions.create(**kwargs)
                else:
                    raise

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


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek provider (OpenAI-compatible API)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config.get('deepseek', {}).get('model', 'deepseek-chat')
        api_key_env = config.get('deepseek', {}).get('api_key_env', 'DEEPSEEK_API_KEY')
        self.api_key = os.getenv(api_key_env)

        if not self.api_key:
            raise ValueError(f"Missing API key: {api_key_env} not set in environment")

        # Import OpenAI SDK (DeepSeek is compatible)
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com",
                timeout=self.timeout
            )
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")

    def call(self, prompt: str, json_mode: bool = False) -> Dict[str, Any]:
        """Call DeepSeek API."""
        try:
            # Configure request
            kwargs = {
                'model': self.model,
                'messages': [{"role": "user", "content": prompt}],
                'temperature': 0.3,
                'max_tokens': 1024,  # DeepSeek still uses max_tokens
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
                'provider': 'deepseek'
            }

        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
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
        elif self.provider_name == 'deepseek':
            self.provider = DeepSeekProvider(config)
        else:
            raise ValueError(f"Unknown provider: {self.provider_name}")

        logger.info(f"Initialized LLM provider: {self.provider_name}")

        # Cost tracking
        self.total_calls = 0
        self.total_cost = 0.0
        self.hourly_calls = {}  # timestamp -> count

        # Per-minute rate limiting (for Gemini free tier: 15 RPM)
        self.max_rpm = config.get('max_requests_per_minute', 15)
        self.minute_requests = deque()  # Track request timestamps in last 60s

    def _wait_for_rate_limit(self):
        """Wait if we're at the per-minute rate limit."""
        current_time = time.time()

        # Clean old requests (older than 60 seconds)
        while self.minute_requests and current_time - self.minute_requests[0] > 60:
            self.minute_requests.popleft()

        # Check if we're at the limit
        if len(self.minute_requests) >= self.max_rpm:
            oldest_request = self.minute_requests[0]
            wait_time = 60 - (current_time - oldest_request) + 0.1  # Add 0.1s buffer

            if wait_time > 0:
                logger.warning(
                    f"Rate limit: {len(self.minute_requests)}/{self.max_rpm} RPM. "
                    f"Waiting {wait_time:.1f}s..."
                )
                time.sleep(wait_time)

                # Clean again after waiting
                current_time = time.time()
                while self.minute_requests and current_time - self.minute_requests[0] > 60:
                    self.minute_requests.popleft()

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

        # Check hourly/daily rate limits
        if not self._check_rate_limits():
            logger.warning("Rate limit exceeded, skipping LLM call")
            return {'success': False, 'error': 'Rate limit exceeded'}

        # Wait if we're at per-minute rate limit
        self._wait_for_rate_limit()

        # Attempt API call with retries
        for attempt in range(retries + 1):
            response = self.provider.call(prompt, json_mode=json_mode)

            if response['success']:
                # Track request timestamp for rate limiting
                self.minute_requests.append(time.time())
                self._track_call()
                return response

            # Check if it's a rate limit error
            error_msg = response.get('error', '')
            if 'quota' in error_msg.lower() or 'rate limit' in error_msg.lower():
                # Parse retry-after time from error message
                retry_after = self._parse_retry_after(error_msg)
                if retry_after:
                    logger.warning(f"Rate limit hit. Retrying after {retry_after:.1f}s...")
                    time.sleep(retry_after)
                    continue

            if attempt < retries:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Retry {attempt + 1}/{retries} after {wait_time}s...")
                time.sleep(wait_time)

        # All retries failed
        logger.error("All LLM API retries failed")
        return {'success': False, 'error': 'Max retries exceeded'}

    def _parse_retry_after(self, error_msg: str) -> Optional[float]:
        """
        Parse retry-after time from error message.

        Args:
            error_msg: Error message from API

        Returns:
            Retry time in seconds, or None if not found
        """
        # Try to parse "retry in X.XXs" or "retry in Xs"
        match = re.search(r'retry in ([\d.]+)s', error_msg, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass

        # Try to parse "retry after X seconds"
        match = re.search(r'retry after ([\d.]+)\s*seconds?', error_msg, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass

        return None

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
        # Clean old minute requests for accurate stats
        current_time = time.time()
        recent_requests = [ts for ts in self.minute_requests if current_time - ts <= 60]

        return {
            'provider': self.provider_name,
            'total_calls': self.total_calls,
            'estimated_cost': self.total_cost,
            'hourly_calls': sum(self.hourly_calls.values()),
            'rpm_current': len(recent_requests),
            'rpm_limit': self.max_rpm
        }
