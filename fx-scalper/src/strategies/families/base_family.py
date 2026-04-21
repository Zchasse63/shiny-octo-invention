"""Base class + shared types for signal families."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class FamilySignals:
    """Boolean entry signals produced by a family.

    Attributes:
        entries_long: True where a long entry should open.
        entries_short: True where a short entry should open.
    """

    entries_long: pd.Series
    entries_short: pd.Series


class SignalFamily(ABC):
    """Abstract interface for a signal family.

    Families are stateless — the same instance runs on any price frame.
    Entry logic only. Exits come from :mod:`src.strategies.exits`.
    """

    name: str  # Override in subclass.

    @abstractmethod
    def generate(self, candles: pd.DataFrame) -> FamilySignals:
        """Compute entries given closed-bar candles.

        Args:
            candles: DataFrame with mid_ / bid_ / ask_ OHLCV columns,
                indexed by tz-aware UTC timestamp. All bars must be
                complete (the caller has already filtered forming bars).

        Returns:
            :class:`FamilySignals` — two boolean series aligned to the
            input index.
        """

    @abstractmethod
    def param_grid(self) -> dict[str, list[Any]]:
        """Return this family's param grid for the Phase-2 sweep.

        Keys are parameter names. Values are the candidate lists.
        """
