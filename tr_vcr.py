"""
Trend Ratio (TR) and Variance Contribution Ratio (VCR).

Per Kris Abdelmessih, "how a high implied vol can be cheap"
(moontowermeta.com, May 2026).

    TR  = RV_sampled_weekly / RV_sampled_daily   over a rolling window
    VCR = max(r²) / sum(r²)                      over the same window

Conventions:
    - Log returns
    - Zero-mean estimator (variance = mean(r²), no demean)
    - Annualization factor = 252 (Kris uses 251; cancels in TR ratio either way)
    - "Weekly" RV uses non-overlapping 5-day blocks ending at t
      (so window=20, sample_freq=5 → 4 blocks covering days [t-19..t])

All functions return pd.Series/DataFrame indexed by the input's date index.
The value at time t uses returns through t close → safe to trade at t+1 open
without look-ahead. The one place look-ahead can leak in is regime
classification via medians/quintiles — use expanding versions for live
backtests (see `expanding_quintile`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------- core estimators ----------

def log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1))


def rv_daily(returns: pd.Series, window: int = 20, ann_factor: int = 252) -> pd.Series:
    """Annualized RV from daily-sampled returns, zero-mean estimator."""
    return np.sqrt((returns ** 2).rolling(window).mean() * ann_factor)


def rv_sampled(prices: pd.Series, window: int = 20, sample_freq: int = 5,
               ann_factor: int = 252) -> pd.Series:
    """
    Annualized RV from k-day non-overlapping returns over a rolling window.
    The k-day return at t is log(P_t / P_{t-k}); we take the n_blocks most
    recent non-overlapping such returns (n_blocks = window // sample_freq).
    """
    if window % sample_freq != 0:
        raise ValueError(f"window ({window}) must be divisible by sample_freq ({sample_freq})")
    n_blocks = window // sample_freq
    log_p = np.log(prices)
    r_k = log_p.diff(sample_freq)              # k-day log return at every t
    sq = r_k ** 2
    sum_sq = sum(sq.shift(i * sample_freq) for i in range(n_blocks))
    return np.sqrt(sum_sq / n_blocks * ann_factor / sample_freq)


def trend_ratio(prices: pd.Series, window: int = 20, sample_freq: int = 5,
                ann_factor: int = 252) -> pd.Series:
    """TR = weekly-sampled RV / daily-sampled RV."""
    r = log_returns(prices)
    return (rv_sampled(prices, window, sample_freq, ann_factor)
            / rv_daily(r, window, ann_factor))


def vcr(returns: pd.Series, window: int = 20) -> pd.Series:
    """VCR = max(r²) / sum(r²) over rolling window."""
    sq = returns ** 2
    return sq.rolling(window).max() / sq.rolling(window).sum()


# ---------- one-shot builder ----------

def features(prices: pd.Series, window: int = 20, sample_freq: int = 5,
             ann_factor: int = 252) -> pd.DataFrame:
    """Return DataFrame of [rv_daily, rv_weekly, tr, vcr] for a single price series."""
    r = log_returns(prices)
    rv_d = rv_daily(r, window, ann_factor)
    rv_w = rv_sampled(prices, window, sample_freq, ann_factor)
    return pd.DataFrame({
        "rv_daily":  rv_d,
        "rv_weekly": rv_w,
        "tr":        rv_w / rv_d,
        "vcr":       vcr(r, window),
    })


# ---------- regime classification ----------

def classify_regime(tr: pd.Series, vcr_: pd.Series,
                    tr_thresh: float | None = None,
                    vcr_thresh: float | None = None) -> pd.Series:
    """
    Four-quadrant classifier (grinding_trend, spike_trend, choppy_grind,
    spike_revert). Splits on medians by default. For live backtests, pass
    explicit thresholds derived from a training/burn-in sample to avoid
    look-ahead.
    """
    if tr_thresh  is None: tr_thresh  = tr.median()
    if vcr_thresh is None: vcr_thresh = vcr_.median()

    high_tr  = tr  > tr_thresh
    high_vcr = vcr_ > vcr_thresh
    out = pd.Series(index=tr.index, dtype="object")
    out[ high_tr & ~high_vcr] = "grinding_trend"
    out[ high_tr &  high_vcr] = "spike_trend"
    out[~high_tr & ~high_vcr] = "choppy_grind"
    out[~high_tr &  high_vcr] = "spike_revert"
    return out


def expanding_quintile(s: pd.Series, min_periods: int = 252,
                       n_bins: int = 5) -> pd.Series:
    """
    Look-ahead-safe rank: at each t, assign s_t to its quantile bin (1..n_bins)
    using only data through t. Slow-ish (rolling apply) — fine for daily
    backtests, cache the result if you re-run often.
    """
    def _rank(window):
        try:
            bins = pd.qcut(window, n_bins, labels=False, duplicates="drop")
            return bins[-1] + 1
        except ValueError:
            return np.nan
    return s.expanding(min_periods=min_periods).apply(_rank, raw=True)


# ---------- usage example ----------

if __name__ == "__main__":
    import yfinance as yf

    px = yf.download("EWY", period="3y", auto_adjust=True)["Close"].squeeze()
    feats = features(px, window=20, sample_freq=5)
    feats["regime"] = classify_regime(feats["tr"], feats["vcr"])
    feats["tr_q"]   = expanding_quintile(feats["tr"])
    print(feats.tail(10))
    print("\nRegime distribution:")
    print(feats["regime"].value_counts(dropna=False))
