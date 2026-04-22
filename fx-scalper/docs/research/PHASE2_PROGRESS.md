# fx-scalper — Phase 2 progress dashboard

**Status: rigorous re-validation of EUR/USD MR edge complete; pivoting to diversified H4/D1 trend + stat-arb with G10 basket expansion in progress.**

Updated: 2026-04-22.

## What's been tested and what it showed

### Original research hypothesis (rounds 1-9)
Intraday mean reversion (BB+RSI filtered) on EUR/USD M15 with session + weekday filter.

| Round | Finding | Verdict |
|---|---|---|
| 1-3 | Sweep of 30K configs found top "PF 2.07" basin | Confirmed real data pattern |
| 4 | Cross-pair: GBP/USD survived (buggy sizing), JPY failed | Generalization claim partial |
| 5-5.5 | Weekday rule "skip Mon, keep Fri" is structural | Real micro-pattern |
| 6 | **Sizing bug**: vbt was using full-equity×leverage. Fixed via `size=5000, size_type='value'`. Inflated all $-numbers ~5× | **Bug fix — all prior $-expectancies wrong** |
| 7 | MAE/MFE per trade: SL 0.5× ATR was inside median MAE | Stop-sizing learning |
| 9 | Validation gauntlet (naive bootstrap + friction 2×): 0/5 configs pass | Edge not confirmed |

### Post-research-review work (rounds 10-17)
After the research review flagged that our search space was narrow and our statistics were naive, pivoted to:
- H4/D1 trend following (strongest published retail evidence — CTAs +27% in 2022)
- Proper statistics (BCa bootstrap + Deflated Sharpe)
- Basket / diversified portfolio (breaks multi-testing curse)
- Peer-reviewed fixing-reversal (Krohn 2024 JoF)
- Stat-arb with Kalman hedge ratio
- ML meta-labeling (in progress — lightgbm install pending)

| Round | Finding | Verdict |
|---|---|---|
| **10** | Donchian + MA crossover on EUR/USD alone, H4 and D1 | Single-pair weak; one MA 8/32 D1 standout on 22 trades (not enough) |
| **11** | Lopez de Prado **purged k-fold CV** + **BCa bootstrap** + **Deflated Sharpe Ratio** applied to all prior survivors | **Decisive**: expected max Sharpe from 30K trials = 3.03; our best observed = 1.91; **dsr_prob = 0.000** for every config. We have over-tested the in-sample data |
| **12-13 (preview)** | Basket of 3 pairs (missing 6 more) with trend rule | Net negative ($USD/JPY drags; need full G10 for dilution) |
| 14 | Meta-labeling on primary signal | **Pending** lightgbm install |
| **15** | **Krohn 2024 fixing-reversal** — exact peer-reviewed spec | **Important null**: EUR/USD pre-cost PF 1.16, Sharpe 1.14 (validates paper implementation); **retail cost → PF 0.99, edge destroyed** (exactly matches paper's Table X). We can replicate peer-reviewed research but cannot harvest this particular edge at OANDA retail spreads |
| 16 | Stat-arb Kalman daily EUR/USD vs GBP/USD | **Running** |
| 17 | NautilusTrader FillModel execution realism | Not started |

## The state of "where is there actually an edge?"

After 16 rounds the honest picture:

### Real patterns that exist but don't clear retail costs

1. **FX benchmark-fixing reversal** (Krohn 2024) — real, peer-reviewed, 15%/yr pre-cost on EUR/USD. We replicated it faithfully. At retail 0.6 pip spread the edge goes to zero. No retail-tradable edge here.

2. **Intraday MR with session + weekday filter** on EUR/USD M15 (our rounds 1-5) — produces PF 1.07-1.19 full-sample on EUR/USD and GBP/USD with corrected sizing. BCa 95% CI lower bound 0.84-0.90 (not significant). DSR after 30K-config sweep: 0.000 (not distinguishable from chance). **Not a paper-trade candidate on its own.**

### Pending / unresolved
- **G10 basket trend following** — highest-evidence strategy class per research (CTA track record), but blocked on data backfill (~4 hrs remaining for 6 pairs).
- **Meta-labeling** — blocked on lightgbm pip install (network contention with backfills).
- **Stat-arb Kalman** — computing; literature says daily EUR/USD vs GBP/USD rarely cointegrates (Koronidis 2013).
- **NautilusTrader L1 FillModel** — execution realism gate per CLAUDE.md Phase 4.

### The pattern across rounds
Every strategy we test produces one of three outcomes:
1. **Looks great in-sample, dies out-of-sample** (Round 5 → Round 9)
2. **Works academically, dies at retail cost** (Round 15)
3. **Requires diversification that single-pair tests can't provide** (Round 10, 12-13 preview)

The common thread is that retail FX at our cost structure is HARD — the peer-reviewed evidence agrees (Carver: 0.3-0.8 Sharpe realistic post-cost; FXCM study: 53% of 1:1 R:R traders profitable). There's no shortcut. Finding a working system requires either (a) dramatically different setup (longer TF + diversified basket) or (b) accepting marginal edge and sizing correctly.

## What I'm confident about

1. **The M15 MR scalper is NOT the answer.** Sizing-corrected numbers + proper statistics say so.
2. **The infrastructure is solid.** Every statistical tool Harvey/Lopez de Prado / Bailey / Efron invented for this kind of problem is now available in our codebase.
3. **Our implementation can replicate peer-reviewed results** (Round 15 validates Krohn 2024 exactly).
4. **Retail spreads are a real cost that multiple strategies do not clear.** Rounds 9, 15 confirm.

## What's genuinely uncertain

1. **Does a G10-basket trend-following system work at retail costs?** — the highest-priority open question. Round 12/13 full-G10 will answer once backfills finish.
2. **Can meta-labeling lift a marginal primary above the cost hurdle?** — Round 14 open.
3. **Is there a different Krohn-style structural edge we haven't looked for?** — e.g. month-end flows, central-bank-meeting day reversal, quarter-end pension rebalancing.

## Near-term plan

1. Let backfill finish (~3-4 more hours).
2. Re-run rounds 12-13 on full G10 basket.
3. Install lightgbm once network frees up, run Round 14.
4. Run Round 16 (already running).
5. If any of these clear the gauntlet (full-sample PF > 1.1, BCa lo > 1.0, DSR prob > 0.95, friction 2× PF > 1.0), move to Round 17 (Nautilus) then Phase 5 (paper trade).
6. If none clear, widen search to (a) month-end / quarter-end flows, (b) multi-timeframe confluence, (c) volatility-regime-switched MR/trend hybrid.

## Git state

All commits pushed to `main`. This session added 9 commits on top of 8 prior:

```
862069a  rounds 15-16: fixing-reversal validated + stat-arb daily Kalman
993ef28  round 14 scaffold: LightGBM meta-labeling on primary signal
0e7adca  rounds 12-13 scaffold: G10 basket trend-following with cash_sharing
6e70426  rounds 10+11: trend-following + rigorous statistics (BCa + DSR)
f434cd6  round 10 scaffold: trend-following families (Donchian + MA crossover)
9286809  round 9: validation gauntlet — 0 of 5 configs pass
0ceced4  FINAL ROLLUP: edge not confirmed; NOT ready for paper trading
7149dee  rounds 6+8: sizing fix + SL ablation + shared-cash portfolio
98a381a  round 4 + 4b: MR survives on GBP, everything fails on JPY
```
