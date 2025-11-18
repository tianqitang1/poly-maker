# OG Maker Bot

Original market making strategy from commit `a6f87c5765088be03aac5a45c202be4644089de8`.

## Strategy Characteristics

**Simpler, More Aggressive Approach:**
- Simple position sizing: keep buying while `position < max_size`
- Stop-loss sells only `trade_size` amount (partial exit, not full position)
- No reverse position checks in buy logic (allows balanced market making)
- Fixed take-profit pricing (no market trend tracking)
- More liberal entry conditions

## Setup

Add credentials to `.env`:
```bash
OG_MAKER_PK=your_private_key_here
OG_MAKER_BROWSER_ADDRESS=your_wallet_address_here
```

## Running

```bash
python og_maker/main.py
```

## Comparison with Too Clever Maker

| Feature | OG Maker | Too Clever Maker |
|---------|----------|------------------|
| Position sizing | Simple < max_size | 0.95 threshold, cancels at 90% |
| Stop-loss | Partial (trade_size) | Full position exit |
| Reverse position check | Only after buy logic | Early check prevents buying |
| Take-profit | Fixed price | Dynamic with trend tracking |
| Entry aggressiveness | More aggressive | More defensive |

## Use Case

Best for:
- Testing if simpler is better
- Markets where balanced two-sided making is profitable
- When you want more aggressive position building
