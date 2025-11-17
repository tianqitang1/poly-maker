"""
Spike Detector

Monitors price movements and detects significant spikes that may indicate
trading opportunities.

Integrates with existing WebSocket infrastructure in poly_data.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
from datetime import datetime, timedelta

from poly_utils.logging_utils import get_logger

logger = get_logger('spike_momentum.spike_detector')


class PriceHistory:
    """Tracks price history for a market."""

    def __init__(self, market_id: str, max_history: int = 1000):
        self.market_id = market_id
        self.max_history = max_history

        # Store (timestamp, price) tuples
        self.history = deque(maxlen=max_history)

        # Current best bid/ask
        self.best_bid = None
        self.best_ask = None
        self.mid_price = None

    def update(self, bid: float, ask: float, timestamp: Optional[float] = None):
        """Update price history."""
        if timestamp is None:
            timestamp = time.time()

        self.best_bid = bid
        self.best_ask = ask
        self.mid_price = (bid + ask) / 2 if bid and ask else None

        if self.mid_price:
            self.history.append((timestamp, self.mid_price))

    def get_price_change(self, window_seconds: int) -> Optional[Tuple[float, float]]:
        """
        Get price change over a time window.

        Args:
            window_seconds: Time window in seconds

        Returns:
            Tuple of (price_change_pct, previous_price) or None
        """
        if not self.history or len(self.history) < 2:
            return None

        current_time = time.time()
        cutoff_time = current_time - window_seconds

        # Find price at start of window
        previous_price = None
        for timestamp, price in self.history:
            if timestamp >= cutoff_time:
                # This is the first price in our window
                previous_price = price
                break

        if previous_price is None or self.mid_price is None:
            return None

        # Calculate percentage change
        if previous_price == 0:
            return None

        price_change_pct = ((self.mid_price - previous_price) / previous_price) * 100

        return (price_change_pct, previous_price)

    def get_volatility(self, window_seconds: int = 300) -> float:
        """
        Calculate price volatility (standard deviation) over window.

        Args:
            window_seconds: Time window in seconds

        Returns:
            Standard deviation of prices
        """
        if not self.history or len(self.history) < 3:
            return 0.0

        current_time = time.time()
        cutoff_time = current_time - window_seconds

        # Get prices in window
        prices = [price for timestamp, price in self.history if timestamp >= cutoff_time]

        if len(prices) < 3:
            return 0.0

        # Calculate standard deviation
        mean_price = sum(prices) / len(prices)
        variance = sum((p - mean_price) ** 2 for p in prices) / len(prices)
        return variance ** 0.5


class Spike:
    """Represents a detected price spike."""

    def __init__(
        self,
        market_id: str,
        market_question: str,
        direction: str,  # 'up' or 'down'
        current_price: float,
        previous_price: float,
        price_change_pct: float,
        time_window: int,
        spike_strength: float,
        detected_at: float
    ):
        self.market_id = market_id
        self.market_question = market_question
        self.direction = direction
        self.current_price = current_price
        self.previous_price = previous_price
        self.price_change_pct = price_change_pct
        self.time_window = time_window
        self.spike_strength = spike_strength  # How many standard deviations
        self.detected_at = detected_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'market_id': self.market_id,
            'market_question': self.market_question,
            'direction': self.direction,
            'current_price': self.current_price,
            'previous_price': self.previous_price,
            'price_change_pct': self.price_change_pct,
            'time_window': self.time_window,
            'spike_strength': self.spike_strength,
            'detected_at': self.detected_at,
            'age_seconds': time.time() - self.detected_at
        }

    def __repr__(self) -> str:
        return (
            f"<Spike: {self.market_question[:50]} "
            f"{self.direction} {self.price_change_pct:+.1f}% "
            f"(strength={self.spike_strength:.1f})>"
        )


class SpikeDetector:
    """Detects price spikes across markets."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize spike detector.

        Args:
            config: Spike detection configuration from config.yaml
        """
        self.config = config
        self.enabled = config.get('enabled', True)

        # Detection parameters
        self.price_change_threshold = config.get('price_change_threshold', 0.02)  # 2%
        self.time_windows = config.get('time_windows', [30, 60, 300])  # seconds
        self.volatility_multiplier = config.get('volatility_multiplier', 2.0)

        # Price history for each market
        self.price_histories: Dict[str, PriceHistory] = {}

        # Recently detected spikes (to avoid duplicates)
        self.recent_spikes: List[Spike] = []
        self.spike_cooldown = 60  # seconds before detecting same market again

        logger.info(
            f"Initialized SpikeDetector "
            f"(threshold={self.price_change_threshold*100:.1f}%, "
            f"windows={self.time_windows}s)"
        )

    def update_price(
        self,
        market_id: str,
        market_question: str,
        best_bid: float,
        best_ask: float
    ) -> Optional[Spike]:
        """
        Update price for a market and check for spikes.

        Args:
            market_id: Market identifier
            market_question: Market question text
            best_bid: Current best bid price
            best_ask: Current best ask price

        Returns:
            Spike object if detected, None otherwise
        """
        if not self.enabled:
            return None

        # Get or create price history
        if market_id not in self.price_histories:
            self.price_histories[market_id] = PriceHistory(market_id)

        history = self.price_histories[market_id]

        # Update price
        history.update(best_bid, best_ask)

        # Check for spikes across all time windows
        for window in self.time_windows:
            spike = self._check_spike(
                market_id=market_id,
                market_question=market_question,
                history=history,
                window_seconds=window
            )

            if spike:
                # Check cooldown to avoid duplicate detections
                if not self._is_on_cooldown(market_id):
                    self.recent_spikes.append(spike)
                    self._cleanup_old_spikes()

                    logger.info(f"Spike detected: {spike}")
                    return spike

        return None

    def _check_spike(
        self,
        market_id: str,
        market_question: str,
        history: PriceHistory,
        window_seconds: int
    ) -> Optional[Spike]:
        """Check if there's a spike in the given time window."""

        # Get price change
        change_data = history.get_price_change(window_seconds)
        if not change_data:
            return None

        price_change_pct, previous_price = change_data

        # Check if absolute change exceeds threshold
        abs_change_pct = abs(price_change_pct)
        if abs_change_pct < self.price_change_threshold * 100:
            return None

        # Calculate volatility
        volatility = history.get_volatility(window_seconds * 2)  # Use 2x window for volatility

        # Calculate spike strength (how many standard deviations)
        if volatility > 0:
            spike_strength = abs(price_change_pct / 100) / volatility
        else:
            spike_strength = abs_change_pct  # If no volatility, use absolute change

        # Check if spike is significant vs normal volatility
        if spike_strength < self.volatility_multiplier:
            return None

        # Spike detected!
        direction = 'up' if price_change_pct > 0 else 'down'

        spike = Spike(
            market_id=market_id,
            market_question=market_question,
            direction=direction,
            current_price=history.mid_price,
            previous_price=previous_price,
            price_change_pct=price_change_pct,
            time_window=window_seconds,
            spike_strength=spike_strength,
            detected_at=time.time()
        )

        return spike

    def _is_on_cooldown(self, market_id: str) -> bool:
        """Check if market is on cooldown (recently detected)."""
        current_time = time.time()
        cutoff_time = current_time - self.spike_cooldown

        for spike in self.recent_spikes:
            if spike.market_id == market_id and spike.detected_at > cutoff_time:
                return True

        return False

    def _cleanup_old_spikes(self):
        """Remove old spikes from recent list."""
        current_time = time.time()
        cutoff_time = current_time - (self.spike_cooldown * 2)

        self.recent_spikes = [
            spike for spike in self.recent_spikes
            if spike.detected_at > cutoff_time
        ]

    def get_recent_spikes(self, max_age_seconds: int = 300) -> List[Spike]:
        """Get recently detected spikes."""
        current_time = time.time()
        cutoff_time = current_time - max_age_seconds

        return [
            spike for spike in self.recent_spikes
            if spike.detected_at > cutoff_time
        ]

    def get_market_history(self, market_id: str) -> Optional[PriceHistory]:
        """Get price history for a market."""
        return self.price_histories.get(market_id)


def test_spike_detector():
    """Test the spike detector."""
    config = {
        'enabled': True,
        'price_change_threshold': 0.02,  # 2%
        'time_windows': [30, 60],
        'volatility_multiplier': 2.0
    }

    detector = SpikeDetector(config)

    # Simulate normal price movements
    print("Simulating normal price movements...")
    for i in range(10):
        detector.update_price(
            market_id='test_market_1',
            market_question='Will the Lakers win?',
            best_bid=0.50 + (i * 0.001),  # Slow drift
            best_ask=0.51 + (i * 0.001)
        )
        time.sleep(0.1)

    # Simulate a spike
    print("\nSimulating price spike...")
    spike = detector.update_price(
        market_id='test_market_1',
        market_question='Will the Lakers win?',
        best_bid=0.65,  # Jump from ~0.51 to 0.65
        best_ask=0.66
    )

    if spike:
        print(f"\nSpike detected!")
        print(f"  Direction: {spike.direction}")
        print(f"  Change: {spike.price_change_pct:+.2f}%")
        print(f"  Strength: {spike.spike_strength:.2f} std devs")
        print(f"  Window: {spike.time_window}s")
    else:
        print("\nNo spike detected (may need more history)")


if __name__ == '__main__':
    test_spike_detector()
