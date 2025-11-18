# Too Clever Maker Bot

Current enhanced market making strategy with all defensive improvements.

## Strategy Characteristics

**More Defensive, Complex Approach:**
- Dynamic position sizing with 0.95 threshold, cancels buys at 90% max_size
- Stop-loss sells ENTIRE position (not partial)
- Early reverse position check prevents buying opposite side
- Market trend tracking for dynamic take-profit pricing
- Multiple layers of over-sizing prevention
- Balance/allowance error handling
- Position correction logic

## Setup

Add credentials to `.env`:
```bash
TOO_CLEVER_MAKER_PK=your_private_key_here
TOO_CLEVER_MAKER_BROWSER_ADDRESS=your_wallet_address_here
```

## Running

```bash
python too_clever_maker/main.py
```

## Comparison with OG Maker

| Feature | OG Maker | Too Clever Maker |
|---------|----------|------------------|
| Position sizing | Simple < max_size | 0.95 threshold, cancels at 90% |
| Stop-loss | Partial (trade_size) | Full position exit |
| Reverse position check | Only after buy logic | Early check prevents buying |
| Take-profit | Fixed price | Dynamic with trend tracking |
| Entry aggressiveness | More aggressive | More defensive |
| Complexity | Simpler | More complex |

## Known Issues (Why It Might Feel "Clunky")

1. **Too Defensive on Entries**: Early reverse position checks prevent balanced market making
2. **Sell-Side Complexity**: Market trend tracking can cause missed fills when market reverses
3. **Over-Sizing Paranoia**: Multiple max_size checks create "dead zones" where bot won't trade
4. **Aggressive Stop-Loss**: Selling entire position on temporary dips prevents recovery

## Use Case

Best for:
- Testing if defensive logic improves performance
- Risk-averse market making
- Markets with high volatility
- Comparing against simpler OG strategy
