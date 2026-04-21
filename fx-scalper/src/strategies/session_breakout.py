"""Strategy 3: London–NY overlap range breakout — scaffold for Day 6.

Per CLAUDE.md §Starter Strategies #3:

* Pair: GBP/USD primary, EUR/USD secondary
* Mark London session range: 08:00–12:00 UTC
* Trade breakout during overlap: 12:00–16:00 UTC
* Fixed 2:1 reward:risk, hard time exit at 16:00 UTC

This module is implemented Day 6 — kept as a stub now so the package layout
is complete and imports resolve.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.strategies.base import Signal, Strategy


@dataclass(frozen=True, slots=True)
class SessionBreakoutParams:
    """Placeholder; filled in Day 6."""

    london_open_utc_hour: int = 8
    london_close_utc_hour: int = 12
    overlap_end_utc_hour: int = 16
    reward_risk_ratio: float = 2.0


class SessionBreakout(Strategy):
    """Day 6 — not yet implemented."""

    NAME = "session_breakout"

    def __init__(self, params: SessionBreakoutParams | None = None) -> None:
        self._p = params or SessionBreakoutParams()

    @property
    def name(self) -> str:
        return self.NAME

    def generate_signal(
        self,
        *,
        instrument: str,
        candles: pd.DataFrame,
    ) -> Signal | None:
        raise NotImplementedError("Strategy 3 implemented in Day 6.")
