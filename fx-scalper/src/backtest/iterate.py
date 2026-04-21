"""Exploration → AI iteration bridge.

Takes a ``full_results.csv`` from :mod:`src.backtest.explorer` and formats
the top performers (+ the duds) into structured prompts that
:mod:`src.utils.ai_research` can send to vectorbtpro's Knowledge module.

Templates live in :func:`build_prompt` — they're deliberately long and
specific because vbt.chat uses RAG over vbtpro's docs corpus, so more
context = better retrieval.

Usage:

    from src.backtest.iterate import (
        load_results, summarize_top_performers, build_prompt,
    )
    from src.utils.ai_research import ask

    df = load_results("backtest_results/explore_20260421_1530/full_results.csv")
    summary = summarize_top_performers(df, n=10)
    prompt = build_prompt("next_iteration", summary)
    result = ask(prompt, tag="explore_iter_1")
    print(result.answer)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

PromptKind = Literal[
    "next_iteration",
    "diagnose_drawdown",
    "compare_families",
    "propose_new_family",
    "explain_anomaly",
]


@dataclass(frozen=True, slots=True)
class TopPerformersSummary:
    """Structured summary of a sweep's top performers.

    Attributes:
        by_profit_factor: Top N rows sorted by profit factor.
        by_win_rate: Top N by win rate.
        by_total_return: Top N by total return.
        worst_drawdowns: Top N by max drawdown (ascending — worst first).
        family_averages: Per-family mean of each metric.
        notes: Human-authored context to include in prompts.
    """

    by_profit_factor: pd.DataFrame
    by_win_rate: pd.DataFrame
    by_total_return: pd.DataFrame
    worst_drawdowns: pd.DataFrame
    family_averages: pd.DataFrame
    notes: str = ""


def load_results(path: str | Path) -> pd.DataFrame:
    """Load the flat exploration results CSV.

    Returns the DataFrame with only OOS rows (IS rows are dropped — we
    report OOS for decisions).
    """
    df = pd.read_csv(path)
    if "kind" in df.columns:
        df = df[df["kind"] == "OOS"].copy()
    return df


def summarize_top_performers(
    df: pd.DataFrame,
    *,
    n: int = 10,
    notes: str = "",
) -> TopPerformersSummary:
    """Collapse a flat results DataFrame into top-N slices per metric.

    Aggregates across walk-forward splits (mean of each metric per family +
    params + exit_config), then ranks.
    """
    # Aggregate across splits.
    key_cols = ["family", "family_params", "exit_config"]
    if not all(c in df.columns for c in key_cols):
        raise ValueError(f"CSV missing required columns: {key_cols}")

    metric_cols = [
        "profit_factor",
        "win_rate",
        "sharpe",
        "sortino",
        "total_trades",
        "max_drawdown_pct",
        "expectancy_usd",
    ]
    agg = (
        df.groupby(key_cols)[metric_cols]
        .mean()
        .reset_index()
    )

    by_pf = agg.sort_values("profit_factor", ascending=False).head(n)
    by_wr = agg.sort_values("win_rate", ascending=False).head(n)
    # Total return isn't a column we explicitly compute — use expectancy × trades as proxy.
    agg["approx_total_return_usd"] = agg["expectancy_usd"] * agg["total_trades"]
    by_ret = agg.sort_values("approx_total_return_usd", ascending=False).head(n)
    worst_dd = agg.sort_values("max_drawdown_pct", ascending=False).head(n)

    family_avg = agg.groupby("family")[metric_cols].mean().reset_index()

    return TopPerformersSummary(
        by_profit_factor=by_pf,
        by_win_rate=by_wr,
        by_total_return=by_ret,
        worst_drawdowns=worst_dd,
        family_averages=family_avg,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


_PROMPTS: dict[PromptKind, str] = {
    "next_iteration": """\
I just ran an exploratory backtest sweep using vectorbt Pro on EUR/USD M1 bars,
2023-01-01 to 2026-04-20, with a common exit framework (SL at N×ATR, optional
fixed TP at N×R, optional ATR/chandelier trail) across six signal families.

**Goal profile:** high-frequency scalping (5+ trades/day), base-hit win rate
plus occasional trailing-stop runners for asymmetric upside. Small account
($500) with 50:1 leverage, $100 margin per trade ($5,000 notional).

**Top 10 configs by profit factor (OOS, walk-forward):**
```
{top_pf}
```

**Top 10 by win rate (OOS):**
```
{top_wr}
```

**Per-family averages across all runs:**
```
{fam_avg}
```

**Notes from the operator:** {notes}

Given this data, draw on vectorbt Pro's documentation and examples to answer:

1. What PATTERNS do you see in the top performers? Any common parameter
   regions (short EMAs? wide stops? trail-off configs?) that correlate
   with strong results?
2. What is MISSING from this sweep? Which signal families, indicator
   combinations, or exit structures should we add for the next iteration?
3. Which vbt Pro features would materially improve the quality of this
   exploration? (Splitter variants? parameter optimization with Optuna?
   vbt.ranges / vbt.trades accessors I should be using?)
4. Any red flags in these numbers that suggest overfitting, data leaks,
   or miscalibration (e.g. win rates or profit factors that look too good
   to be real at this frequency)?

Please be specific — reference vbt modules / function names / example
notebook patterns where applicable.
""",
    "diagnose_drawdown": """\
One of my top-ranked strategies just had a bad equity-curve period. Here's
the relevant metric context:

**Strategy:**
```
{strategy_desc}
```

**Walk-forward OOS rows (all splits):**
```
{rows}
```

**Observed drawdown period characteristics:** {notes}

Using vectorbt Pro's documentation and Discord archive, answer:

1. What's the most likely cause of drawdown clustering like this on minute-
   bar FX data?
2. Which vbt Pro drawdown / trades accessors should I query to diagnose
   further? Specific function names and usage examples.
3. What filters or regime detection patterns might mitigate this without
   overfitting?
""",
    "compare_families": """\
I have six signal families under test. Here are the per-family aggregate
metrics (mean across all param combos × exit configs × OOS splits):

```
{fam_avg}
```

Using vbt Pro's knowledge corpus, answer:

1. Which family is strongest overall, and why? Reference the metric
   distribution, not just the mean.
2. Are there FAMILY PAIRS that look complementary (different regimes)?
   How would you combine them — via vbt meta-strategy patterns, signal
   overlays, or portfolio-level rotation?
3. Which of these families is most consistent with an FX scalping profile
   per vbtpro's example notebooks and published patterns?

Notes: {notes}
""",
    "propose_new_family": """\
I've tested six signal families (pullback_ema, range_breakout,
vwap_deviation, ema_cross, bb_rsi_mr, rsi_extreme) on EUR/USD M1.
I'm looking for signal families I haven't tried yet that might produce
the "scalping + trail for runners" profile.

Looking at vectorbt Pro's indicator library and strategy examples, propose
3-5 NEW signal families worth implementing next. For each:

- Rough signal logic (entries + exits)
- vbt indicator / pattern / factory reference to build on
- Why it might match the scalping profile
- Expected trade frequency and typical failure mode

Constraint: must be implementable on M1/M5 bid/ask data (no orderbook /
tick-level features beyond what Dukascopy bi5 provides).

Existing families + their current metric profile:

```
{fam_avg}
```
""",
    "explain_anomaly": """\
I see an anomaly in my exploration results. Specifically:

{notes}

**Relevant rows:**
```
{rows}
```

Using vbt Pro's knowledge, explain what could produce this pattern. Be
concrete: if it's a vbt-specific gotcha, cite the function. If it's a
backtesting artifact, explain the mechanism.
""",
}


def build_prompt(
    kind: PromptKind,
    summary: TopPerformersSummary | None = None,
    *,
    strategy_desc: str = "",
    rows: str = "",
    notes: str = "",
) -> str:
    """Format a prompt template with actual data.

    Args:
        kind: Which template to use.
        summary: :class:`TopPerformersSummary` for ``next_iteration`` /
            ``compare_families`` / ``propose_new_family``.
        strategy_desc: Free text description for ``diagnose_drawdown``.
        rows: Formatted rows string for ``diagnose_drawdown`` / ``explain_anomaly``.
        notes: Human context appended to most templates.

    Returns:
        The fully-formatted prompt string.
    """
    tmpl = _PROMPTS[kind]

    def _fmt(df: pd.DataFrame, n: int = 10) -> str:
        if df.empty:
            return "(empty)"
        try:
            return df.head(n).to_markdown(index=False, floatfmt=".4f")
        except Exception:
            return df.head(n).to_string(index=False)

    if summary is not None:
        return tmpl.format(
            top_pf=_fmt(summary.by_profit_factor),
            top_wr=_fmt(summary.by_win_rate),
            top_ret=_fmt(summary.by_total_return),
            worst_dd=_fmt(summary.worst_drawdowns),
            fam_avg=_fmt(summary.family_averages, n=20),
            notes=notes or summary.notes or "(none)",
        )
    return tmpl.format(
        strategy_desc=strategy_desc,
        rows=rows,
        notes=notes or "(none)",
        fam_avg="(not provided)",
        top_pf="(not provided)",
        top_wr="(not provided)",
    )
