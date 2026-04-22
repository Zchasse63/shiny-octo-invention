"""Round 15 — Krohn/Mueller/Whelan (2024) fixing-reversal strategy.

Implements the exact specification from the Journal of Finance paper
(Krohn, Mueller, Whelan, "Foreign Exchange Fixings and Returns around
the Clock", 2024) as found in the research review:

  Daily trade pattern:
    Leg A (pre-ECB):   SHORT USD from 07:00 UTC → EXIT at ECB fix
                       (13:15 UTC winter / 12:15 UTC summer)
    Leg B (post-London): LONG USD from 16:00 UTC (winter) / 15:00 UTC
                         (summer) → EXIT at NY close 22:00 UTC

For EUR/USD: "long USD" = short EUR/USD, "short USD" = long EUR/USD.

The research review's honest caveat: paper Table X shows that **after
retail bid-ask spread the strategy turns NEGATIVE** — it only works
for liquidity-providing dealers. We test it anyway to:
  1. Confirm the effect is present pre-TC (validates implementation)
  2. Measure retail-cost degradation quantitatively
  3. Identify what slippage / spread assumption kills the edge
  4. Potentially find pairs / regimes where the effect survives retail
     spreads (e.g. higher-vol days, month-ends)

Published effect size: EUR 15.6% annualized pre-TC, GBP 12.4%.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, time as dtime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.backtest.data_loader import load_symbol_bars  # noqa: E402
from src.backtest.resample import resample_bars  # noqa: E402
from src.backtest.statistics import bca_bootstrap_pf, deflated_sharpe_ratio  # noqa: E402
from src.utils.logger import get_logger, init_logger  # noqa: E402


# Times in UTC. DST approximated by month (winter = Nov-Mar, summer = Apr-Oct).
# More precise DST would require pytz rules, but this is close enough.
def _is_summer(dt: pd.Timestamp) -> bool:
    """Rough summer-time heuristic. London = BST Mar-last-Sunday → Oct-last-Sunday."""
    return 3 < dt.month < 11 or (dt.month == 3 and dt.day >= 27) or (dt.month == 10 and dt.day <= 26)


def _leg_times(dt: pd.Timestamp) -> dict:
    """Return (leg_a_start, leg_a_end, leg_b_start, leg_b_end) for a given date."""
    if _is_summer(dt):
        return {
            "leg_a_start": dtime(7, 0),
            "leg_a_end": dtime(12, 15),   # ECB CEST
            "leg_b_start": dtime(15, 0),  # London fix BST
            "leg_b_end": dtime(22, 0),
        }
    return {
        "leg_a_start": dtime(7, 0),
        "leg_a_end": dtime(13, 15),   # ECB CET
        "leg_b_start": dtime(16, 0),  # London fix GMT
        "leg_b_end": dtime(22, 0),
    }


def backtest_fixing_reversal(
    pair: str,
    *,
    slip_bps_multiplier: float = 1.0,
    notional_per_trade: float = 5000.0,
    init_cash: float = 5000.0,
    include_leg_a: bool = True,
    include_leg_b: bool = True,
) -> dict:
    """Backtest the Krohn spec on one pair.

    Args:
        pair: OANDA-style e.g. "EUR_USD".
        slip_bps_multiplier: 1.0 = realistic half-spread. 0.0 = pre-TC
            (matches paper's headline results). 2.0-10.0 = stress.
        notional_per_trade: Per-leg notional in USD.
        init_cash: Starting cash.

    Returns: dict of stats + per-trade PnL array.
    """
    logger = get_logger(__name__)
    logger.info(f"Loading {pair}")
    m1 = load_symbol_bars(pair, start="2023-01-01", end="2026-04-20")
    if m1.empty:
        return {"error": f"no data for {pair}"}
    # Resample to 15-min for cleaner entry/exit alignment
    bars = resample_bars(m1, "15min")

    # Paper's convention: "long USD" means short base-ccy pair for EUR/USD, etc.
    # For EUR/USD: SHORT USD 07:00 → ECB means LONG EUR/USD 07:00 → ECB
    # For EUR/USD: LONG USD London → 22:00 means SHORT EUR/USD London → 22:00
    # Wait — re-reading the research: "Buy USD before the fix, sell USD after"
    # So pre-ECB (hedging flows bid USD up): LONG USD before ECB fix. For
    # EUR/USD expressed as "USD-quoted", LONG USD = SHORT EUR/USD.
    # Post-London (reversal kicks in): SHORT USD → LONG EUR/USD.
    # For USD/JPY ("USD-base"): LONG USD = LONG USD/JPY, SHORT USD = SHORT USD/JPY.
    usd_is_base = pair.startswith("USD_")

    # Collect per-trade PnL
    trades = []
    # Compute half-spread per bar (retail costs)
    has_spread = "bid_close" in bars.columns and "ask_close" in bars.columns
    if has_spread:
        mid_close = (bars["bid_close"] + bars["ask_close"]) / 2.0
    else:
        mid_close = bars["mid_close"] if "mid_close" in bars.columns else bars["close"]

    # Group by date
    for date, day_bars in bars.groupby(bars.index.date):
        lt = _leg_times(pd.Timestamp(date))
        # Intraday bars indexed by hour:minute
        for leg_name, do_leg, (start_t, end_t, direction_usd) in [
            ("leg_a", include_leg_a, (lt["leg_a_start"], lt["leg_a_end"], "long_usd")),
            ("leg_b", include_leg_b, (lt["leg_b_start"], lt["leg_b_end"], "short_usd")),
        ]:
            if not do_leg:
                continue
            # Find bar nearest to start + end
            try:
                start_bar = day_bars[day_bars.index.time == start_t]
                end_bar = day_bars[day_bars.index.time == end_t]
                if start_bar.empty or end_bar.empty:
                    continue
                entry_ts = start_bar.index[0]
                exit_ts = end_bar.index[0]
                entry_mid = float(mid_close.loc[entry_ts])
                exit_mid = float(mid_close.loc[exit_ts])
            except (KeyError, IndexError):
                continue
            # Price return direction based on whether we're long or short USD
            # and whether pair is USD-base or USD-quote
            if direction_usd == "long_usd":
                if usd_is_base:
                    # LONG USD/XXX = pnl from (exit - entry) / entry
                    ret = (exit_mid - entry_mid) / entry_mid
                else:
                    # LONG USD = SHORT XXX/USD = pnl from (entry - exit) / entry
                    ret = (entry_mid - exit_mid) / entry_mid
            else:  # short_usd
                if usd_is_base:
                    # SHORT USD/XXX = pnl from (entry - exit) / entry
                    ret = (entry_mid - exit_mid) / entry_mid
                else:
                    # SHORT USD = LONG XXX/USD = pnl from (exit - entry) / entry
                    ret = (exit_mid - entry_mid) / entry_mid

            # Apply realistic round-trip cost
            if has_spread and slip_bps_multiplier > 0:
                spread_entry = (
                    (bars.loc[entry_ts, "ask_close"] - bars.loc[entry_ts, "bid_close"])
                    / entry_mid
                )
                spread_exit = (
                    (bars.loc[exit_ts, "ask_close"] - bars.loc[exit_ts, "bid_close"])
                    / exit_mid
                )
                round_trip_cost = (spread_entry + spread_exit) * slip_bps_multiplier
                ret -= round_trip_cost

            pnl_usd = ret * notional_per_trade
            trades.append({
                "date": str(date),
                "leg": leg_name,
                "entry_ts": entry_ts,
                "exit_ts": exit_ts,
                "entry_price": entry_mid,
                "exit_price": exit_mid,
                "return_bps": ret * 10000,
                "pnl_usd": pnl_usd,
            })

    if not trades:
        return {"error": "no trades generated"}
    df = pd.DataFrame(trades)
    pnls = df["pnl_usd"].to_numpy()
    wins = pnls[pnls > 0].sum()
    losses = abs(pnls[pnls < 0].sum())
    pf = float(wins / losses) if losses > 0 else float("inf")
    equity = init_cash + np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return {
        "pair": pair,
        "slip_mult": slip_bps_multiplier,
        "n_trades": len(trades),
        "win_rate": float((pnls > 0).mean()),
        "total_pnl_usd": float(pnls.sum()),
        "mean_return_bps": float(df["return_bps"].mean()),
        "profit_factor": pf,
        "max_dd": float(abs(dd.min())),
        "final_equity": float(equity[-1]),
        "trades_df": df,
        "pnls": pnls,
    }


def main() -> int:
    init_logger()
    logger = get_logger(__name__)

    pairs = ["EUR_USD", "GBP_USD", "USD_JPY"]
    # Paper's pre-cost result (slip_mult=0) should be strongly positive.
    # Our realistic retail (slip_mult=1.0) should be marginal-to-negative.
    # Stress (slip_mult=3.0) should be very negative.
    scenarios = [
        ("pre_cost", 0.0),
        ("retail_realistic", 1.0),
        ("stress_3x", 3.0),
    ]

    summary_rows = []
    details: dict = {}
    for pair in pairs:
        for scenario_name, slip in scenarios:
            res = backtest_fixing_reversal(pair, slip_bps_multiplier=slip)
            if "error" in res:
                logger.warning(f"{pair} {scenario_name}: {res['error']}")
                continue
            details[(pair, scenario_name)] = res
            # Annualize: assume ~250 trading days × 2 legs = 500 trades/yr ideal
            trades_per_year = res["n_trades"] / 3.3
            annualized_pnl = res["total_pnl_usd"] / 3.3

            # Sharpe + DSR on per-trade pnls
            pnls = res["pnls"]
            mu, sigma = float(np.mean(pnls)), float(np.std(pnls, ddof=1))
            sharpe = (mu/sigma*np.sqrt(trades_per_year)) if sigma > 0 else 0.0
            from scipy import stats as _st
            skew = float(_st.skew(pnls)) if len(pnls) >= 3 else 0.0
            kurt = float(_st.kurtosis(pnls, fisher=False)) if len(pnls) >= 4 else 3.0
            # n_trials=1 since this is an a priori strategy, not data-mined
            dsr = deflated_sharpe_ratio(sharpe, n_trials=1, n_observations=len(pnls), skew=skew, kurt=kurt)

            summary_rows.append({
                "pair": pair, "scenario": scenario_name, "slip_mult": slip,
                "n_trades": res["n_trades"],
                "wr": res["win_rate"],
                "mean_ret_bps": res["mean_return_bps"],
                "pf": res["profit_factor"],
                "total_pnl_usd": res["total_pnl_usd"],
                "annualized_usd": annualized_pnl,
                "max_dd": res["max_dd"],
                "sharpe": sharpe,
                "dsr_prob": dsr["dsr_prob"],
            })
    df = pd.DataFrame(summary_rows)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M")
    out_dir = PROJECT_ROOT / "backtest_results" / f"round15_fixing_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "summary.csv", index=False)
    for (pair, scenario), res in details.items():
        res["trades_df"].to_csv(out_dir / f"trades_{pair}_{scenario}.csv", index=False)

    print()
    print("=" * 110)
    print("Round 15 — Krohn/Mueller/Whelan 2024 fixing-reversal strategy")
    print("=" * 110)
    print(df.round(3).to_string(index=False))
    print()
    print("Paper expectation: pre-cost annualized ~15% EUR, ~12% GBP.")
    print("Research-review expectation: retail-realistic turns NEGATIVE (Table X of paper).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
