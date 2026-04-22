"""Round 5.5 — weekday-neighbor lone-spike test.

Per vbt.chat (docs/research/ai_queries/20260422T155429-...), the headline
finding that ``tue_fri`` beats ``mon_thu`` / ``tue_thu`` / ``all`` could be a
lone spike (multi-testing noise) or part of a smooth manifold (real edge).
The discriminator is to score neighboring weekday presets and see if they
cluster near ``tue_fri``'s PF or collapse.

We pin every other parameter to the round-5 top-1 config
(bb_rsi_mr_filtered, M15, BB(20,2.25), RSI(14,25,75), ADX off,
london_ny_overlap, spread 0.25) and sweep only ``weekday`` across:

    all, tue_thu, mon_thu, tue_fri, wed_fri, tue_wed_fri, mon_tue_fri

If ``tue_fri`` remains distinctly best while ``wed_fri`` and ``tue_wed_fri``
collapse, the signal is likely noise-driven. If ``wed_fri`` / ``tue_wed_fri``
are within a small band of ``tue_fri``, the weekday edge is likely real.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.backtest.data_loader import load_symbol_bars  # noqa: E402
from src.backtest.explorer import ExploreConfig, explore  # noqa: E402
from src.backtest.resample import resample_bars  # noqa: E402
from src.strategies.families.filtered_mr import (  # noqa: E402
    FilteredBBRSIMRFamily,
    FilteredBBRSIMRParams,
    _WEEKDAY_PRESETS,
)
from src.utils.logger import get_logger, init_logger  # noqa: E402


WEEKDAYS = ["all", "tue_thu", "mon_thu", "tue_fri",
            "wed_fri", "tue_wed_fri", "mon_tue_fri"]


class RoundFiveFiveFamily(FilteredBBRSIMRFamily):
    """Pinned-param variant of FilteredBBRSIMR that only varies weekday.

    Used for round 5.5 lone-spike discriminator — cheaper than re-running
    the full round-5 grid because the only axis is the weekday filter.
    """

    name = "bb_rsi_mr_filtered_r55"
    params_cls = FilteredBBRSIMRParams

    def param_grid(self) -> dict[str, list]:
        # Pin every dimension except weekday to the round-5 top-1 config.
        return {
            "bb_length": [20],
            "bb_std": [2.25],
            "rsi_length": [14],
            "rsi_long_threshold": [25.0],
            "rsi_short_threshold": [75.0],
            "max_adx": [None],
            "session": ["london_ny_overlap"],
            "weekday": WEEKDAYS,
            "max_spread_atr_frac": [0.25],
        }


def main() -> int:
    init_logger()
    logger = get_logger(__name__)

    logger.info("Round 5.5: loading EUR/USD M1 bars 2023-01..2026-04")
    m1 = load_symbol_bars("EUR_USD", start="2023-01-01", end="2026-04-20")
    m15 = resample_bars(m1, "15min")
    logger.info(f"Resampled to M15: {len(m15):,} bars")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M")
    out_dir = PROJECT_ROOT / "backtest_results" / f"explore_round5_5_{stamp}"

    cfg = ExploreConfig(
        data_range=(m15.index.min().isoformat(), m15.index.max().isoformat()),
        walk_forward_windows=3,
        train_frac=0.5,
        random_subset_per_family=None,   # take every combo (only 7)
        exit_random_subset=None,         # every exit config
        initial_cash=500.0,
    )

    df = explore(m15, [RoundFiveFiveFamily()], cfg, output_dir=out_dir)

    # Rank per weekday on OOS mean PF.
    oos = df[df["kind"] == "OOS"].copy()
    oos["weekday"] = oos["family_params"].str.extract(r'"weekday": "(\w+)"')
    ranked = (
        oos.groupby("weekday")[
            ["profit_factor", "expectancy_usd", "win_rate", "max_drawdown_pct",
             "total_trades"]
        ]
        .mean()
        .sort_values("profit_factor", ascending=False)
    )
    print()
    print("=" * 76)
    print("Round 5.5 — weekday neighbor lone-spike test (OOS mean across splits)")
    print("=" * 76)
    print(ranked.round(3).to_string())
    print()
    ranked.to_csv(out_dir / "weekday_ranking.csv")
    logger.info(f"Wrote ranking to {out_dir / 'weekday_ranking.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
