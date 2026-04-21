# Round 3.5 — meta-analysis: capital, circuit breakers, portfolio

All derived from existing round-3 data — no new backtests. Addresses the
question "what other factors can we tune besides indicators and strategies?"

## 1. Capital scaling changes which strategies are survivable

Our current CLAUDE.md circuit breakers are tuned for a $500 account:
- Account floor: $400 (20% DD halts bot)
- Daily loss limit: $50 (10% daily halt)
- Single-trade max loss: $30 (6% halt)

**The round-3 winners ALL fail the $400 floor.** Their observed max-drawdowns (during walk-forward OOS) exceed $100, putting the trough below $400:

| Timeframe | Family | Best config DD$ (on $500) | Trough |
|---|---|---|---|
| M5 | rsi_extreme_filtered | $130 | $370 (below $400 floor) |
| M15 | bb_rsi_mr | $193 | $307 |
| M30 | bb_rsi_mr_filtered | $172 | $328 |
| H1 | ema_cross | $214 | $286 |

At the current $400 floor, every winner would halt mid-run.

## 2. Larger capital → the same dollar DD becomes a smaller % of account

Scaling starting capital from $500 → $2000 while keeping position size constant ($100 margin/trade = $5000 notional):

| TF | Family | DD$ | Trough @$500 | Trough @$1k | Trough @$1.5k | Trough @$2k |
|---|---|---|---|---|---|---|
| M5 | rsi_extreme_filtered | $130 | **$370** | $870 | $1370 | $1870 |
| M15 | bb_rsi_mr | $193 | **$307** | $807 | $1307 | $1807 |
| M30 | bb_rsi_mr_filtered | $172 | **$328** | $828 | $1328 | $1828 |
| H1 | ema_cross | $214 | **$286** | $786 | $1286 | $1786 |

**At $1500 capital all top configs hold above a $400 floor** with comfortable cushion. At $2000 they all hold above a $750 floor.

## 3. Circuit breaker × capital survival matrix

Whether each top config survives for different (capital, fixed-$ floor) combos:

```
M5 rsi_extreme_filtered (DD $130):
              $200  $300  $400  $500  $750  $1000
  Cap $500    ✓     ✓     ✗     ✗     ✗     ✗
  Cap $1000   ✓     ✓     ✓     ✓     ✓     ✗
  Cap $1500   ✓     ✓     ✓     ✓     ✓     ✓
  Cap $2000   ✓     ✓     ✓     ✓     ✓     ✓

H1 ema_cross (DD $214 — worst):
              $200  $300  $400  $500  $750  $1000
  Cap $500    ✓     ✗     ✗     ✗     ✗     ✗
  Cap $1000   ✓     ✓     ✓     ✓     ✓     ✗
  Cap $1500   ✓     ✓     ✓     ✓     ✓     ✓
```

## 4. BUT a proportional floor kills everything

If we follow the "standard" risk-management rule that `floor = 75% of starting cap` (not fixed dollars), the DD% becomes the binding constraint — and that doesn't scale with capital:

| Config | DD % | Survives 25% DD limit? |
|---|---|---|
| M5 rsi_extreme_filtered | 26% | **NO** (marginal) |
| M15 bb_rsi_mr | 38.6% | NO |
| M30 bb_rsi_mr_filtered | 34.4% | NO |
| H1 ema_cross | 42.8% | NO |

**Takeaway:** capital scaling fixes *absolute* drawdown survivability but not *percentage* drawdown. If our philosophy is "halt when we've lost 25% of the account, regardless of size," NONE of round-3 winners pass. If our philosophy is "halt if we drop below a specific dollar floor," more capital makes it easier.

Suggested CB policy:
- Keep an **absolute dollar floor** proportional to initial capital BUT scaled to tolerate observed DDs. At $1500 cap, $1100 floor = 27% DD tolerance. All round-3 top configs would survive their observed max.
- Keep the daily loss limit at **~10% of account** (scales with capital).
- Keep the consecutive-loss pause and single-trade blowout at **percentages of account**, not dollars.

## 5. Multi-strategy portfolio — running M5 + M15 + M30 simultaneously

If you allocate $500 to each of the top M5, M15, M30 configs on a $1500 account:

**Returns (annualized, assuming walk-forward OOS holds):**
- Combined trades per 8-month split: 198
- Weighted $/trade: **$6.78**
- Annualized profit: **+$2,013**
- ROI on $1500: **+134%**

**Drawdown depends on correlation between strategies:**
- Perfectly correlated (worst case): $495 → 33% of $1500
- Uncorrelated (best case): $289 → 19%
- Realistic (corr ~0.4): $297 → **20%**

**Single-strategy M5 on all $1500 (for comparison):**
- Annualized: +$1,092 (+73%)
- Max DD: $389 (26%)

The portfolio **beats single-strategy on both return and risk** IF strategies are truly uncorrelated. Round 4 cross-pair testing will reveal whether these 3 signal families are independent or just rhyming with each other.

## 6. Return/Risk-Ratio (RRR) rankings

Annual profit divided by max drawdown, on $500 starting capital:

| TF | Family | Annual $ | DD $ | **RRR** |
|---|---|---|---|---|
| M5 | rsi_extreme_filtered | $364 | $130 | 2.80× |
| **M15** | **bb_rsi_mr** | **$1,237** | **$193** | **6.41×** |
| M30 | bb_rsi_mr_filtered | $411 | $172 | 2.39× |
| H1 | ema_cross | $774 | $214 | 3.61× |

**M15 `bb_rsi_mr` dominates on risk-adjusted basis.** 6.4× RRR means you earn 6.4× your peak drawdown per year. That's exceptional (professional standard is >1.5×; >3× is great).

## Concrete proposed configurations for round 4+ testing

### Option A — aggressive single-strategy (best expected return)
- Capital: $1500
- Strategy: M15 bb_rsi_mr (unfiltered)
- Config: `BB(30, 2.5), RSI(21), long_threshold=35, short_threshold=80, SL 0.5×ATR, TP 1.5R, Chandelier trail 2×ATR`
- Expected annual: ~$3,711 (+247% ROI) with ~$580 max DD (39%)
- CB settings: $900 floor (60% of $1500), 15% daily loss limit
- Risk: trips own DD floor if backtest OOS doesn't hold

### Option B — portfolio diversification (best risk-adjusted)
- Capital: $1500, divided $500 per strategy
- Strategies: M5 rsi_extreme_filtered + M15 bb_rsi_mr + M30 bb_rsi_mr_filtered (top config of each)
- Expected annual: ~$2,013 (+134% ROI) with ~$297 max DD (20%)
- CB settings: $1200 floor (80% of $1500), 10% daily loss limit per strategy
- Risk: portfolio correlation may be higher than assumed, eroding DD benefit

### Option C — defensive (if cross-pair round 4 degrades)
- Capital: $1000
- Strategy: M5 rsi_extreme_filtered only (lowest DD of the four)
- Expected annual: ~$728 (+73% ROI) with ~$130 max DD (13% of $1000)
- CB settings: $750 floor (75% of $1000), 10% daily loss limit
- Most conservative; best chance of surviving if OOS degrades

## Caveats

1. **DD correlation assumption is unverified.** I used an $\sqrt{N}$ bound for portfolio DD. Real correlation could be higher (if all strategies bleed in the same months) or lower. Proper measurement needs per-trade equity curves, which we don't have saved yet — that's a backtest infrastructure gap for round 5.

2. **"Annualized" assumes walk-forward OOS performance holds forward.** In reality, retail strategies often degrade by 30-50% going from backtest to paper to live. Scale all projected annuals by ~0.5 as a honest base case.

3. **$100 margin/trade scales notional linearly.** If you double cap from $500 to $1000 and keep $100/trade, you have half the % exposure per trade. To get the projected ROI scaling above, you'd need to ALSO scale position size proportionally.

4. **Sharpe in round-3 reports was wrong** (commit `7fdaf66` fixed it). Use PF, expectancy, and DD as primary metrics for now; Sortino is also reliable because it's a ratio that doesn't depend on annualization.
