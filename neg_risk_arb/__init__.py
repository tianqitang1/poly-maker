"""
Negative Risk Arbitrage Bot for Polymarket

A specialized arbitrage bot that exploits negative risk markets where
YES + NO < $1.00, executing both sides and merging positions for instant profit.

Features:
- Automated scanning for arbitrage opportunities
- Atomic execution with partial fill protection
- Configurable risk management via config.yaml
- Real-time profit tracking
"""

__version__ = "1.0.0"
