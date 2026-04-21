"""Thin wrapper over ``pandas-ta-classic`` with consistent column names.

Per CLAUDE.md §Day 2: pandas-ta-classic is the primary indicator API; TA-Lib
is a silent speed booster underneath. Callers use functions in *this* module
only — never import pandas_ta_classic directly from strategies.

Column-name conventions (all lowercase, snake_case, stable across versions):

* ``rsi_{n}``
* ``bb_upper_{n}_{s}``, ``bb_middle_{n}_{s}``, ``bb_lower_{n}_{s}``
* ``atr_{n}``
* ``adx_{n}``
* ``ema_{n}``
* ``macd_line``, ``macd_signal``, ``macd_hist`` (pandas-ta defaults 12/26/9)
* ``stoch_k_{k}_{d}_{sm}``, ``stoch_d_{k}_{d}_{sm}``
* ``supertrend_{n}_{m}``, ``supertrend_dir_{n}_{m}``

All functions expect a DataFrame with a ``mid_close``/``mid_high``/``mid_low``
columns (or plain ``close``/``high``/``low``) and return a copy with new
columns appended.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
import pandas_ta_classic as ta

# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------

def _oc(df: pd.DataFrame, which: str) -> pd.Series:
    """Return high/low/close/open — preferring mid_ columns but falling back."""
    mid_col = f"mid_{which}"
    if mid_col in df.columns:
        return df[mid_col]
    if which in df.columns:
        return df[which]
    # Also try bid-based fallback.
    bid_col = f"bid_{which}"
    if bid_col in df.columns:
        return df[bid_col]
    raise KeyError(f"Required column missing: {mid_col!r} or {which!r} or {bid_col!r}")


def talib_available() -> bool:
    """True if pandas-ta-classic detected TA-Lib and is using it under the hood.

    CLAUDE.md §Day 2 validation test.

    In pandas-ta-classic 0.4.x ``Imports`` is a dict mapping optional-package
    names to a boolean flag (e.g. ``{'talib': True, 'numba': False, ...}``).
    We handle both dict and attribute access defensively in case a future
    release changes the shape.
    """
    try:
        from pandas_ta_classic import Imports  # type: ignore[attr-defined]

        if isinstance(Imports, dict):
            return bool(Imports.get("talib", False))
        return bool(getattr(Imports, "talib", False))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Individual indicators
# ---------------------------------------------------------------------------

def add_rsi(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """Append ``rsi_{length}`` column."""
    out = df.copy()
    out[f"rsi_{length}"] = ta.rsi(_oc(out, "close"), length=length)
    return out


def add_bbands(
    df: pd.DataFrame,
    length: int = 20,
    std: float = 2.0,
) -> pd.DataFrame:
    """Append Bollinger Bands: upper/middle/lower + bandwidth."""
    out = df.copy()
    bb = ta.bbands(_oc(out, "close"), length=length, std=std)
    if bb is None or bb.empty:
        out[f"bb_lower_{length}_{std}"] = pd.NA
        out[f"bb_middle_{length}_{std}"] = pd.NA
        out[f"bb_upper_{length}_{std}"] = pd.NA
        return out
    # pandas-ta columns are formatted e.g. "BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0".
    lower_col = next(c for c in bb.columns if c.startswith("BBL_"))
    middle_col = next(c for c in bb.columns if c.startswith("BBM_"))
    upper_col = next(c for c in bb.columns if c.startswith("BBU_"))
    out[f"bb_lower_{length}_{std}"] = bb[lower_col]
    out[f"bb_middle_{length}_{std}"] = bb[middle_col]
    out[f"bb_upper_{length}_{std}"] = bb[upper_col]
    return out


def add_atr(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """Append ``atr_{length}`` column."""
    out = df.copy()
    out[f"atr_{length}"] = ta.atr(
        high=_oc(out, "high"),
        low=_oc(out, "low"),
        close=_oc(out, "close"),
        length=length,
    )
    return out


def add_adx(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """Append ``adx_{length}`` column (ignore +DI/-DI for now)."""
    out = df.copy()
    adx = ta.adx(
        high=_oc(out, "high"),
        low=_oc(out, "low"),
        close=_oc(out, "close"),
        length=length,
    )
    if adx is None or adx.empty:
        out[f"adx_{length}"] = pd.NA
        return out
    adx_col = next(c for c in adx.columns if c.startswith("ADX_"))
    out[f"adx_{length}"] = adx[adx_col]
    return out


def add_ema(df: pd.DataFrame, length: int = 200) -> pd.DataFrame:
    """Append ``ema_{length}`` column."""
    out = df.copy()
    out[f"ema_{length}"] = ta.ema(_oc(out, "close"), length=length)
    return out


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Append ``macd_line``, ``macd_signal``, ``macd_hist`` columns."""
    out = df.copy()
    macd = ta.macd(_oc(out, "close"), fast=fast, slow=slow, signal=signal)
    if macd is None or macd.empty:
        out["macd_line"] = pd.NA
        out["macd_signal"] = pd.NA
        out["macd_hist"] = pd.NA
        return out
    line_col = next(c for c in macd.columns if c.startswith("MACD_") and "h" not in c and "s" not in c)
    signal_col = next(c for c in macd.columns if c.startswith("MACDs_"))
    hist_col = next(c for c in macd.columns if c.startswith("MACDh_"))
    out["macd_line"] = macd[line_col]
    out["macd_signal"] = macd[signal_col]
    out["macd_hist"] = macd[hist_col]
    return out


def add_stoch(
    df: pd.DataFrame,
    k: int = 14,
    d: int = 3,
    smooth_k: int = 3,
) -> pd.DataFrame:
    """Append stochastic %K and %D columns."""
    out = df.copy()
    stoch = ta.stoch(
        high=_oc(out, "high"),
        low=_oc(out, "low"),
        close=_oc(out, "close"),
        k=k,
        d=d,
        smooth_k=smooth_k,
    )
    suffix = f"{k}_{d}_{smooth_k}"
    if stoch is None or stoch.empty:
        out[f"stoch_k_{suffix}"] = pd.NA
        out[f"stoch_d_{suffix}"] = pd.NA
        return out
    k_col = next(c for c in stoch.columns if c.startswith("STOCHk_"))
    d_col = next(c for c in stoch.columns if c.startswith("STOCHd_"))
    out[f"stoch_k_{suffix}"] = stoch[k_col]
    out[f"stoch_d_{suffix}"] = stoch[d_col]
    return out


def add_supertrend(
    df: pd.DataFrame,
    length: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """Append supertrend level + direction."""
    out = df.copy()
    st = ta.supertrend(
        high=_oc(out, "high"),
        low=_oc(out, "low"),
        close=_oc(out, "close"),
        length=length,
        multiplier=multiplier,
    )
    suffix = f"{length}_{multiplier}"
    if st is None or st.empty:
        out[f"supertrend_{suffix}"] = pd.NA
        out[f"supertrend_dir_{suffix}"] = pd.NA
        return out
    lvl_col = next(c for c in st.columns if c.startswith("SUPERT_") and not c.startswith("SUPERTd_"))
    dir_col = next(c for c in st.columns if c.startswith("SUPERTd_"))
    out[f"supertrend_{suffix}"] = st[lvl_col]
    out[f"supertrend_dir_{suffix}"] = st[dir_col]
    return out


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def add_indicators(
    df: pd.DataFrame,
    *,
    indicators: Iterable[str] = ("rsi", "bbands", "atr", "adx"),
) -> pd.DataFrame:
    """Run several indicators at once with default parameters.

    Args:
        df: Input frame with mid_/bid_/ask_ OHLCV columns.
        indicators: Names to compute. Defaults to the set used by Strategy 1.

    Returns:
        Copy of ``df`` with indicator columns appended.
    """
    out = df.copy()
    for name in indicators:
        match name:
            case "rsi":
                out = add_rsi(out)
            case "bbands":
                out = add_bbands(out)
            case "atr":
                out = add_atr(out)
            case "adx":
                out = add_adx(out)
            case "ema":
                out = add_ema(out)
            case "macd":
                out = add_macd(out)
            case "stoch":
                out = add_stoch(out)
            case "supertrend":
                out = add_supertrend(out)
            case _:
                raise ValueError(f"Unknown indicator: {name!r}")
    return out
