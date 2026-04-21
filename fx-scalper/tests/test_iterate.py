"""Tests for the exploration → prompt iteration bridge."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.backtest.iterate import (
    TopPerformersSummary,
    build_prompt,
    load_results,
    summarize_top_performers,
)


@pytest.fixture
def synthetic_results() -> pd.DataFrame:
    """A miniature flat results frame covering 2 families × 3 param combos × 2 splits."""
    rows = []
    for family in ("pullback_ema", "range_breakout"):
        for i in range(3):
            for split in ("win1", "win2"):
                for kind in ("IS", "OOS"):
                    rows.append(
                        {
                            "family": family,
                            "family_params": f'{{"p": {i}}}',
                            "exit_config": f'{{"e": {i}}}',
                            "split": split,
                            "kind": kind,
                            "profit_factor": 1.0 + i * 0.3 + (0.1 if kind == "OOS" else 0),
                            "win_rate": 0.5 + i * 0.05,
                            "sharpe": 0.4 + i * 0.2,
                            "sortino": 0.6 + i * 0.2,
                            "total_trades": 50 + i * 10,
                            "max_drawdown_pct": 0.15 - i * 0.02,
                            "expectancy_usd": 0.5 * (i + 1),
                            "calmar": 0.3,
                            "cagr": 0.1,
                            "avg_drawdown_duration_bars": 20,
                        }
                    )
    return pd.DataFrame(rows)


def test_load_results_drops_is_rows(
    synthetic_results: pd.DataFrame, tmp_path: Path
) -> None:
    csv = tmp_path / "results.csv"
    synthetic_results.to_csv(csv, index=False)
    df = load_results(csv)
    assert (df["kind"] == "OOS").all()


def test_summarize_top_performers_produces_ranked_frames(
    synthetic_results: pd.DataFrame,
) -> None:
    oos = synthetic_results[synthetic_results["kind"] == "OOS"]
    summary = summarize_top_performers(oos, n=5)
    assert isinstance(summary, TopPerformersSummary)
    assert not summary.by_profit_factor.empty
    assert summary.by_profit_factor.iloc[0]["profit_factor"] >= summary.by_profit_factor.iloc[-1][
        "profit_factor"
    ]
    # Aggregation collapsed splits: 2 families × 3 combos = 6 unique rows.
    assert len(summary.family_averages) == 2


def test_summarize_requires_key_columns(
    synthetic_results: pd.DataFrame,
) -> None:
    bad = synthetic_results.drop(columns=["family"])
    with pytest.raises(ValueError, match="missing required columns"):
        summarize_top_performers(bad)


def test_build_prompt_next_iteration_contains_context(
    synthetic_results: pd.DataFrame,
) -> None:
    oos = synthetic_results[synthetic_results["kind"] == "OOS"]
    summary = summarize_top_performers(oos, n=5)
    prompt = build_prompt("next_iteration", summary, notes="Smoke test")
    assert "EUR/USD" in prompt
    assert "profit_factor" in prompt
    assert "Smoke test" in prompt
    # Families should appear in the family_averages block.
    assert "pullback_ema" in prompt
    assert "range_breakout" in prompt


def test_build_prompt_diagnose_drawdown_no_summary_needed() -> None:
    prompt = build_prompt(
        "diagnose_drawdown",
        summary=None,
        strategy_desc="pullback_ema with fast=20, slow=50, sl=1.5×ATR, trail=atr×2",
        rows="split1: DD 18%, split2: DD 22%, split3: DD 25%",
        notes="DD clustered around end-of-month 2024-06",
    )
    assert "pullback_ema" in prompt
    assert "2024-06" in prompt
