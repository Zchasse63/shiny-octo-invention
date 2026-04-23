"""Round 14 — LightGBM meta-labeling on the MR primary signal.

Per the research review (Lopez de Prado AFML Ch. 3; final vbt.chat
response), meta-labeling is probably the highest-ROI ML application
on existing retail FX primary signals. The idea:

  1. Take an existing primary signal (our bb_rsi_mr_filtered).
  2. For every primary signal, hand-label whether the trade hit its
     profit target (+1) or stop (-1) first — "triple barrier" method.
  3. Train a secondary classifier (LightGBM) on features at the entry
     bar to predict the probability of a profitable outcome.
  4. Only take primary signals where the classifier's probability
     exceeds a threshold. This filters low-conviction trades.

Expected lift per research: 0.1-0.3 Sharpe improvement on the primary,
with no new entry logic. Implemented with:

  - Purged k-fold CV (Lopez de Prado) — prevents label-overlap leakage
  - Triple-barrier labels matching our SL/TP framework
  - Feature set: rolling vol, ATR regime, time-of-day, cross-asset
    proxies (VIX-like realized-vol), lag returns, session dummies

If the meta-labeler's precision > 0.75 on the held-out fold AND PF
lifts materially vs the unfiltered primary, this is a real edge.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import vectorbtpro as vbt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, precision_score, recall_score, roc_auc_score,
)

from src.backtest.data_loader import load_symbol_bars  # noqa: E402
from src.backtest.resample import resample_bars  # noqa: E402
from src.backtest.statistics import (  # noqa: E402
    bca_bootstrap_pf,
    deflated_sharpe_ratio,
    purged_kfold_splits,
)
from src.indicators.engine import add_adx, add_atr  # noqa: E402
from src.strategies.families.filtered_mr import (  # noqa: E402
    FilteredBBRSIMRFamily,
    FilteredBBRSIMRParams,
)
from src.utils.logger import get_logger, init_logger  # noqa: E402


# The primary signal we're meta-labeling: Round-5 top-1 filtered config
PRIMARY_PARAMS = FilteredBBRSIMRParams(
    bb_length=40, bb_std=2.5, rsi_length=21,
    rsi_long_threshold=20.0, rsi_short_threshold=70.0,
    max_adx=None, session="active", weekday="tue_fri",
    max_spread_atr_frac=0.5,
)


def _build_triple_barrier_labels(
    bars: pd.DataFrame,
    entries_long: pd.Series,
    entries_short: pd.Series,
    atr: pd.Series,
    *,
    sl_atr_mult: float = 2.0,
    tp_r_mult: float = 1.5,
    max_bars_held: int = 96,  # 24 hours on M15
) -> pd.DataFrame:
    """Generate triple-barrier labels for every primary signal.

    For each entry, walk forward bar-by-bar and check:
      - did price hit +TP × ATR first? label = +1
      - did price hit -SL × ATR first? label = -1
      - did we run out of time (max_bars_held)? label = 0

    Returns a DataFrame with 'entry_bar', 'direction', 'label',
    'exit_bar', 'return_bps'.
    """
    close_col = "mid_close" if "mid_close" in bars.columns else "bid_close"
    high_col = "mid_high" if "mid_high" in bars.columns else "bid_high"
    low_col = "mid_low" if "mid_low" in bars.columns else "bid_low"
    close = bars[close_col].to_numpy()
    highs = bars[high_col].to_numpy()
    lows = bars[low_col].to_numpy()

    records = []
    for direction, entries in [("long", entries_long), ("short", entries_short)]:
        for i, is_entry in enumerate(entries.to_numpy()):
            if not is_entry:
                continue
            entry_price = close[i]
            a = atr.iloc[i] if i < len(atr) else float("nan")
            if not np.isfinite(a) or a <= 0:
                continue
            sl_dist = sl_atr_mult * a
            tp_dist = tp_r_mult * sl_dist

            label = 0
            exit_bar = min(i + max_bars_held, len(close) - 1)
            exit_price = close[exit_bar]
            for j in range(i + 1, min(i + max_bars_held + 1, len(close))):
                if direction == "long":
                    if highs[j] >= entry_price + tp_dist:
                        label = 1; exit_bar = j; exit_price = entry_price + tp_dist; break
                    if lows[j] <= entry_price - sl_dist:
                        label = -1; exit_bar = j; exit_price = entry_price - sl_dist; break
                else:
                    if lows[j] <= entry_price - tp_dist:
                        label = 1; exit_bar = j; exit_price = entry_price - tp_dist; break
                    if highs[j] >= entry_price + sl_dist:
                        label = -1; exit_bar = j; exit_price = entry_price + sl_dist; break
            ret_bps = ((exit_price - entry_price) / entry_price) * 10000 \
                * (1 if direction == "long" else -1)
            records.append({
                "entry_bar": i, "direction": direction,
                "label": label, "exit_bar": exit_bar,
                "return_bps": ret_bps, "bars_held": exit_bar - i,
                "entry_price": entry_price,
            })
    return pd.DataFrame(records)


def _build_features(
    bars: pd.DataFrame, atr: pd.Series, adx: pd.Series, rsi: pd.Series,
) -> pd.DataFrame:
    """Meta-labeler feature set — purely price/vol-derived, no news.

    Kept small (15ish features) since with ~600 trades on 3 yrs we
    don't have the data for a wide feature set.
    """
    close_col = "mid_close" if "mid_close" in bars.columns else "bid_close"
    close = bars[close_col]
    ret = close.pct_change()

    feats = pd.DataFrame(index=bars.index)
    # Volatility regime
    feats["atr_norm"] = atr / close
    feats["atr_zscore_60"] = (atr - atr.rolling(60).mean()) / atr.rolling(60).std()
    # Realized vol
    feats["rvol_20"] = ret.rolling(20).std()
    feats["rvol_pct_rank_252"] = feats["rvol_20"].rolling(252).rank(pct=True)
    # Momentum / trend
    feats["ret_1"] = ret
    feats["ret_5"] = close.pct_change(5)
    feats["ret_20"] = close.pct_change(20)
    feats["ma_ratio_fast_slow"] = close.ewm(span=8).mean() / close.ewm(span=32).mean() - 1
    # Oscillators
    feats["rsi_val"] = rsi
    feats["rsi_slope_5"] = rsi.diff(5)
    feats["adx_val"] = adx
    # Time features
    feats["hour"] = bars.index.hour
    feats["weekday"] = bars.index.weekday
    # Session dummies
    feats["is_overlap"] = feats["hour"].between(12, 15).astype(int)
    feats["is_active"] = feats["hour"].between(7, 16).astype(int)
    # Bar-shape features
    high_col = "mid_high" if "mid_high" in bars.columns else "bid_high"
    low_col = "mid_low" if "mid_low" in bars.columns else "bid_low"
    hl_range = (bars[high_col] - bars[low_col]) / close
    feats["hl_range"] = hl_range
    feats["hl_range_zscore_60"] = (hl_range - hl_range.rolling(60).mean()) / hl_range.rolling(60).std()

    return feats


def main() -> int:
    init_logger()
    logger = get_logger(__name__)

    # Prefer lightgbm; fall back to sklearn HistGradientBoostingClassifier
    # which is pure Python and doesn't need libomp (avoids install pain).
    try:
        import lightgbm as lgb  # noqa: F401
        USE_LGBM = True
    except (ImportError, OSError):
        from sklearn.ensemble import HistGradientBoostingClassifier
        USE_LGBM = False
        logger.info("lightgbm unavailable; using sklearn HistGradientBoostingClassifier")

    # Load EUR/USD M15 (our primary signal's native TF)
    # Default to full range; override with FXSCALPER_START / FXSCALPER_END env.
    import os
    start = os.environ.get("FXSCALPER_START", "2023-01-01")
    end = os.environ.get("FXSCALPER_END", "2026-04-20")
    logger.info(f"Loading EUR/USD M15 data {start}..{end}")
    m1 = load_symbol_bars("EUR_USD", start=start, end=end)
    m15 = resample_bars(m1, "15min")
    close_col = "mid_close" if "mid_close" in m15.columns else "bid_close"
    close = m15[close_col]

    # Primary signal
    primary = FilteredBBRSIMRFamily(PRIMARY_PARAMS)
    sigs = primary.generate(m15)
    logger.info(f"Primary signal entries: long={int(sigs.entries_long.sum())} "
                f"short={int(sigs.entries_short.sum())}")

    # Indicators
    atr = add_atr(m15, length=14)["atr_14"]
    adx = add_adx(m15, length=14)["adx_14"]
    # RSI via pandas-ta
    import pandas_ta as ta
    rsi = ta.rsi(close, length=14)
    assert rsi is not None
    # Triple-barrier labels
    logger.info("Building triple-barrier labels (SL=2x ATR, TP=1.5R)")
    labels = _build_triple_barrier_labels(
        m15, sigs.entries_long, sigs.entries_short, atr,
        sl_atr_mult=2.0, tp_r_mult=1.5, max_bars_held=96,
    )
    logger.info(f"  {len(labels)} labeled primary signals")
    logger.info(f"  label distribution: +1={int((labels['label']==1).sum())} "
                f"-1={int((labels['label']==-1).sum())} "
                f"0={int((labels['label']==0).sum())}")
    # Primary-alone win rate:
    primary_pnl = labels["return_bps"].to_numpy()
    primary_wr = float((primary_pnl > 0).mean())
    primary_expectancy = float(primary_pnl.mean())
    logger.info(f"Primary alone: WR={primary_wr:.1%}  expectancy={primary_expectancy:.2f} bps")

    # Features at entry bar
    feats_all = _build_features(m15, atr, adx, rsi)
    X = feats_all.iloc[labels["entry_bar"].to_numpy()].copy().reset_index(drop=True)
    # Binary label: win (+1 / >0) vs not
    y = (labels["return_bps"] > 0).astype(int).to_numpy()
    # Add direction as a feature
    X["direction_long"] = (labels["direction"] == "long").astype(int).to_numpy()

    X_clean = X.fillna(0.0)

    # Purged k-fold CV, k=5
    logger.info("Running purged 5-fold CV on LightGBM meta-labeler")
    splits = purged_kfold_splits(len(X_clean), k=5, embargo_frac=0.02)
    all_preds = np.full(len(X_clean), np.nan)
    metrics_per_fold = []
    for split in splits:
        tr_X, tr_y = X_clean.iloc[split.train_idx], y[split.train_idx]
        te_X, te_y = X_clean.iloc[split.test_idx], y[split.test_idx]
        if USE_LGBM:
            model = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.05,
                num_leaves=31, max_depth=6, min_child_samples=20,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbose=-1,
            )
        else:
            model = HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05,
                max_depth=6, max_leaf_nodes=31, min_samples_leaf=20,
                random_state=42,
            )
        model.fit(tr_X, tr_y)
        probs = model.predict_proba(te_X)[:, 1]
        all_preds[split.test_idx] = probs
        preds = (probs > 0.5).astype(int)
        metrics_per_fold.append({
            "fold": split.fold,
            "acc": accuracy_score(te_y, preds),
            "prec": precision_score(te_y, preds, zero_division=0),
            "recall": recall_score(te_y, preds, zero_division=0),
            "auc": roc_auc_score(te_y, probs) if len(np.unique(te_y)) > 1 else float("nan"),
            "n_test": len(te_y),
        })
    metrics_df = pd.DataFrame(metrics_per_fold)
    print()
    print("=" * 80)
    print("Round 14 — meta-labeler CV metrics per purged fold")
    print("=" * 80)
    print(metrics_df.round(3).to_string(index=False))

    # Now sweep probability thresholds and see which maximizes meta-labeled PF
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
    print()
    print("=" * 80)
    print("Meta-labeled performance vs probability threshold:")
    print("=" * 80)
    threshold_rows = []
    for t in thresholds:
        mask = all_preds >= t  # take trade only if meta-prob >= t
        if mask.sum() < 10:
            continue
        filtered_pnl = primary_pnl[mask]
        wins_sum = filtered_pnl[filtered_pnl > 0].sum()
        losses_sum = abs(filtered_pnl[filtered_pnl < 0].sum())
        pf = float(wins_sum/losses_sum) if losses_sum > 0 else float("inf")
        threshold_rows.append({
            "threshold": t,
            "n_trades": int(mask.sum()),
            "retained_frac": float(mask.mean()),
            "wr": float((filtered_pnl > 0).mean()),
            "expectancy_bps": float(filtered_pnl.mean()),
            "pf": pf,
            "total_bps": float(filtered_pnl.sum()),
        })
    thresh_df = pd.DataFrame(threshold_rows)
    print(thresh_df.round(3).to_string(index=False))

    # Bootstrap CI on the best threshold
    if not thresh_df.empty:
        best = thresh_df.loc[thresh_df["pf"].idxmax()]
        t_best = float(best["threshold"])
        mask_best = all_preds >= t_best
        bca = bca_bootstrap_pf(primary_pnl[mask_best], n_resamples=5000)
        print()
        print(f"Best threshold: {t_best}")
        print(f"  PF point: {bca.point_estimate:.3f}")
        print(f"  BCa 95% CI: [{bca.ci_lower:.3f}, {bca.ci_upper:.3f}]")
        print(f"  Primary alone: PF {(primary_pnl[primary_pnl>0].sum() / abs(primary_pnl[primary_pnl<0].sum())):.3f}, WR {primary_wr:.1%}")

    # Save outputs
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M")
    out_dir = PROJECT_ROOT / "backtest_results" / f"round14_meta_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_dir / "cv_metrics.csv", index=False)
    thresh_df.to_csv(out_dir / "threshold_sweep.csv", index=False)
    # Save labeled dataset
    labels["meta_prob"] = all_preds
    labels.to_csv(out_dir / "labels_with_meta.csv", index=False)
    logger.info(f"Artifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
