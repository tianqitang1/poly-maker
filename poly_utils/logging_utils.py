"""
Unified Logging System for Poly-Maker Bots

Provides structured logging with:
- Detailed logs (DEBUG level - everything)
- Trade logs (INFO level - only actions/changes)
- Daily summaries
- Log rotation
- Console + file output
"""

import logging
import json
import os
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import numpy as np


class BotLogger:
    """
    Unified logger for all poly-maker bots.

    Creates multiple log streams:
    - Detailed log: Everything (debug level)
    - Trade log: Only significant events (orders, fills, merges, etc.)
    - Summary: Daily summaries
    """

    def __init__(
        self,
        bot_name: str,
        log_dir: str = "logs",
        console_level: str = "INFO",
        file_level: str = "DEBUG"
    ):
        """
        Initialize logger for a specific bot.

        Args:
            bot_name: Name of the bot (e.g., 'main', 'near_sure', 'neg_risk_arb')
            log_dir: Directory for log files
            console_level: Logging level for console output
            file_level: Logging level for file output
        """
        self.bot_name = bot_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create bot-specific subdirectory
        self.bot_log_dir = self.log_dir / bot_name
        self.bot_log_dir.mkdir(exist_ok=True)

        # Initialize loggers
        self.logger = logging.getLogger(f"polymaker.{bot_name}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False  # Don't propagate to root logger

        # Clear existing handlers
        self.logger.handlers.clear()

        # Setup handlers
        self._setup_console_handler(console_level)
        self._setup_detailed_file_handler(file_level)
        self._setup_trade_file_handler()

        # Track trade/action events for summaries
        self.trade_history = []
        self.session_start = datetime.now()

    def _setup_console_handler(self, level: str):
        """Setup console output handler."""
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level))
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

    def _setup_detailed_file_handler(self, level: str):
        """Setup detailed rotating file handler."""
        detailed_log = self.bot_log_dir / f"{self.bot_name}_detailed.log"

        # Rotating file handler - 10MB per file, keep 5 backups
        file_handler = RotatingFileHandler(
            detailed_log,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, level))
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)

    def _setup_trade_file_handler(self):
        """Setup trade-specific file handler (daily rotation)."""
        trade_log = self.bot_log_dir / f"{self.bot_name}_trades.log"

        # Daily rotating file handler
        trade_handler = TimedRotatingFileHandler(
            trade_log,
            when='midnight',
            interval=1,
            backupCount=30  # Keep 30 days
        )
        trade_handler.setLevel(logging.INFO)

        # Create filter to only log trade-related events
        class TradeFilter(logging.Filter):
            def filter(self, record):
                # Only log if message contains trade keywords or is INFO+
                trade_keywords = ['order', 'trade', 'fill', 'merge', 'buy', 'sell',
                                 'cancel', 'stop', 'profit', 'loss', 'arbitrage']
                msg_lower = record.getMessage().lower()
                return any(kw in msg_lower for kw in trade_keywords) or record.levelno >= logging.WARNING

        trade_handler.addFilter(TradeFilter())

        trade_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        trade_handler.setFormatter(trade_format)
        self.logger.addHandler(trade_handler)

    # Convenience methods
    def debug(self, msg: str, **kwargs):
        """Log debug message."""
        self.logger.debug(msg, **kwargs)

    def info(self, msg: str, **kwargs):
        """Log info message."""
        self.logger.info(msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        """Log warning message."""
        self.logger.warning(msg, **kwargs)

    def error(self, msg: str, **kwargs):
        """Log error message."""
        self.logger.error(msg, **kwargs)

    def critical(self, msg: str, **kwargs):
        """Log critical message."""
        self.logger.critical(msg, **kwargs)

    # Trade-specific logging
    def log_order(self, action: str, market: str, token: str, price: float, size: float, **extra):
        """
        Log an order placement.

        Args:
            action: BUY or SELL
            market: Market name/question
            token: Token ID
            price: Order price
            size: Order size
            **extra: Additional fields
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'order',
            'action': action,
            'market': market,
            'token': token,
            'price': price,
            'size': size,
            **extra
        }

        self.trade_history.append(event)
        self.info(f"ORDER {action}: {market[:50]} | {size:.2f} @ ${price:.4f}")
        self._append_to_json_log(event)

    def log_fill(self, action: str, market: str, token: str, fill_price: float, fill_size: float, **extra):
        """Log an order fill."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'fill',
            'action': action,
            'market': market,
            'token': token,
            'fill_price': fill_price,
            'fill_size': fill_size,
            **extra
        }

        self.trade_history.append(event)
        self.info(f"FILL {action}: {market[:50]} | {fill_size:.2f} @ ${fill_price:.4f}")
        self._append_to_json_log(event)

    def log_cancel(self, market: str, token: str, reason: str = "", **extra):
        """Log order cancellation."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'cancel',
            'market': market,
            'token': token,
            'reason': reason,
            **extra
        }

        self.trade_history.append(event)
        self.info(f"CANCEL: {market[:50]} | Reason: {reason}")
        self._append_to_json_log(event)

    def log_merge(self, market: str, size: float, recovered: float, **extra):
        """Log position merge."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'merge',
            'market': market,
            'size': size,
            'recovered': recovered,
            **extra
        }

        self.trade_history.append(event)
        self.info(f"MERGE: {market[:50]} | {size:.2f} shares → ${recovered:.2f}")
        self._append_to_json_log(event)

    def log_stop_loss(self, market: str, position: float, entry_price: float,
                     exit_price: float, pnl: float, reason: str = "", **extra):
        """Log stop loss trigger."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'stop_loss',
            'market': market,
            'position': position,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': (exit_price - entry_price) / entry_price * 100 if entry_price > 0 else 0,
            'reason': reason,
            **extra
        }

        self.trade_history.append(event)
        self.warning(f"STOP LOSS: {market[:50]} | PnL: ${pnl:.2f} | Reason: {reason}")
        self._append_to_json_log(event)

    def log_arbitrage(self, market: str, yes_price: float, no_price: float,
                     size: float, profit: float, success: bool, **extra):
        """Log arbitrage execution."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'arbitrage',
            'market': market,
            'yes_price': yes_price,
            'no_price': no_price,
            'total_cost': yes_price + no_price,
            'size': size,
            'profit': profit,
            'profit_bps': (1 - (yes_price + no_price)) * 10000,
            'success': success,
            **extra
        }

        self.trade_history.append(event)
        if success:
            self.info(f"ARBITRAGE SUCCESS: {market[:50]} | Profit: ${profit:.2f}")
        else:
            self.warning(f"ARBITRAGE FAILED: {market[:50]} | {extra.get('error', 'Unknown error')}")
        self._append_to_json_log(event)

    def _convert_to_serializable(self, obj):
        """
        Convert numpy/pandas types to native Python types for JSON serialization.

        Args:
            obj: Any object that might contain numpy/pandas types

        Returns:
            Object with all numpy/pandas types converted to native Python types
        """
        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # pandas types
            return obj.item()
        else:
            return obj

    def _append_to_json_log(self, event: Dict):
        """Append event to JSON log file."""
        json_log = self.bot_log_dir / f"{self.bot_name}_events.jsonl"

        # Convert numpy/pandas types to native Python types
        serializable_event = self._convert_to_serializable(event)

        with open(json_log, 'a') as f:
            f.write(json.dumps(serializable_event) + '\n')

    def generate_daily_summary(self) -> Dict:
        """
        Generate daily summary from trade history.

        Returns:
            Summary dictionary with stats
        """
        if not self.trade_history:
            return {'message': 'No trades recorded'}

        # Filter today's events
        today = datetime.now().date()
        today_events = [
            e for e in self.trade_history
            if datetime.fromisoformat(e['timestamp']).date() == today
        ]

        # Count event types
        event_counts = {}
        for event in today_events:
            event_type = event['event_type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        # Calculate P&L from fills, merges, and arbitrages
        total_pnl = 0.0

        for event in today_events:
            if event['event_type'] == 'stop_loss':
                total_pnl += event.get('pnl', 0)
            elif event['event_type'] == 'arbitrage' and event.get('success'):
                total_pnl += event.get('profit', 0)
            elif event['event_type'] == 'merge':
                # Merge profit is typically small (from neg risk arb)
                pass

        summary = {
            'date': today.isoformat(),
            'bot': self.bot_name,
            'session_duration_hours': (datetime.now() - self.session_start).total_seconds() / 3600,
            'total_events': len(today_events),
            'event_breakdown': event_counts,
            'total_pnl': round(total_pnl, 2),
            'orders_placed': event_counts.get('order', 0),
            'fills': event_counts.get('fill', 0),
            'cancels': event_counts.get('cancel', 0),
            'merges': event_counts.get('merge', 0),
            'stop_losses': event_counts.get('stop_loss', 0),
            'arbitrages': event_counts.get('arbitrage', 0),
        }

        return summary

    def save_daily_summary(self):
        """Save daily summary to file."""
        summary = self.generate_daily_summary()

        summary_file = self.bot_log_dir / f"{self.bot_name}_daily_summary.jsonl"

        with open(summary_file, 'a') as f:
            f.write(json.dumps(summary) + '\n')

        self.info(f"Daily summary saved: {summary['total_events']} events, P&L: ${summary['total_pnl']:.2f}")

        return summary

    def print_summary(self):
        """Print summary to console."""
        summary = self.generate_daily_summary()

        print(f"\n{'='*80}")
        print(f"DAILY SUMMARY - {self.bot_name.upper()}")
        print(f"{'='*80}")
        print(f"Date: {summary.get('date', 'N/A')}")
        print(f"Session Duration: {summary.get('session_duration_hours', 0):.1f} hours")
        print(f"Total Events: {summary.get('total_events', 0)}")
        print(f"\nEvent Breakdown:")
        for event_type, count in summary.get('event_breakdown', {}).items():
            print(f"  {event_type}: {count}")
        print(f"\nTotal P&L: ${summary.get('total_pnl', 0):.2f}")
        print(f"{'='*80}\n")


def get_logger(bot_name: str, **kwargs) -> BotLogger:
    """
    Get or create a logger for a bot.

    Args:
        bot_name: Name of the bot
        **kwargs: Additional arguments for BotLogger

    Returns:
        BotLogger instance
    """
    return BotLogger(bot_name, **kwargs)
