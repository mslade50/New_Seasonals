"""k1 shared helpers for 2026-09-15 checks C2/C4/C7.

Only the pooled/heterogeneity layer; every per-cell statistic comes from
pitch_lab (declusters, summarize, sign_test, local_control, wilder_atr).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import (declusters, load_prices, local_control, pct_rank,  # noqa
                       sign_test, summarize, wilder_atr)

ASOF = pd.Timestamp("2026-09-14")

# W25's reference class: the 23 of 33 REF names with n>=3 under the W25 rule
# (scratch/pitch_checks/2026-08-27/b2_c7_smh_refclass.txt section 1).
REF23 = ["XLE", "VNQ", "SMH", "IYR", "XHB", "XLI", "KRE", "XME", "XLF", "XOP",
         "QQQ", "XRT", "ITB", "OIH", "FXI", "ITA", "XLB", "IBB", "IWM", "XBI",
         "EEM", "GDX", "XLY"]
REF33 = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV",
         "XLY", "SMH", "IBB", "IHI", "ITA", "ITB", "XBI", "XHB", "XME", "XOP",
         "XRT", "KRE", "OIH", "VNQ", "IYR", "GDX", "QQQ", "IWM", "SPY", "DIA",
         "EEM", "EFA", "FXI"]


def vret(s: pd.Series, n: int) -> pd.Series:
    v = s.dropna()
    return (v / v.shift(n) - 1.0).reindex(s.index)


def fwd(s: pd.Series, h: int, lag: int = 1) -> pd.Series:
    return s.shift(-(lag + h)) / s.shift(-lag) - 1.0


def atr_series(df: pd.DataFrame) -> pd.Series:
    return pd.Series(np.asarray(wilder_atr(df["High"], df["Low"], df["Close"]),
                                dtype=float), index=df.index)


def cell_stats(ret: pd.Series, mask: pd.Series, min_gap: int,
               drift_all: bool = True) -> dict:
    """Declustered episodes of `mask` on `ret` (signal-date aligned)."""
    valid = ret.dropna().index
    m = mask.reindex(ret.index, fill_value=False).fillna(False).astype(bool)
    trig = ret.index[m.values].intersection(valid)
    if len(trig) == 0:
        return {"n": 0, "n_days": 0, "_vals": np.array([]),
                "_dates": pd.DatetimeIndex([])}
    epi = declusters(trig, min_gap, valid)
    v = ret.loc[epi].values.astype(float)
    base = ret.loc[valid].values
    n = len(v)
    sd = v.std(ddof=1) if n > 1 else np.nan
    return {"n_days": len(trig), "n": n, "mean_pct": 100 * v.mean(),
            "drift_pct": 100 * base.mean(),
            "excess_pct": 100 * (v.mean() - base.mean()),
            "se_pct": 100 * sd / np.sqrt(n) if n > 1 else np.nan,
            "hit": 100 * (v > 0).mean(), "worst_pct": 100 * v.min(),
            "_vals": v, "_dates": epi, "_drift": base.mean()}


def cochran(names, excess_pct, se_pct) -> dict:
    from scipy import stats
    d = pd.DataFrame({"t": names, "y": excess_pct, "se": se_pct}).dropna()
    d = d[d.se > 0]
    if len(d) < 2:
        return {}
    w = 1.0 / d.se.values ** 2
    y = d.y.values
    mu = (w * y).sum() / w.sum()
    Q = float((w * (y - mu) ** 2).sum())
    k = len(d)
    p = float(stats.chi2.sf(Q, k - 1))
    I2 = max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0
    # DerSimonian-Laird tau^2 for shrinkage
    c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - (k - 1)) / c) if c > 0 else 0.0
    return {"k": k, "Q": Q, "p": p, "I2": I2, "fe_pct": float(mu),
            "fe_se_pct": float(np.sqrt(1 / w.sum())), "tau2": tau2}


def shrink(member_y: float, member_se: float, fam_mu: float, tau2: float) -> float:
    """Empirical-Bayes posterior mean of one member toward the family mean."""
    if tau2 <= 0:
        return fam_mu
    b = member_se ** 2 / (member_se ** 2 + tau2)
    return b * fam_mu + (1 - b) * member_y


def date_clusters(dates, vals, gap_days: int = 14) -> tuple[np.ndarray, list]:
    """Collapse pooled cross-name episodes whose dates sit within `gap_days`
    CALENDAR days of the previous one into a single averaged cluster."""
    order = np.argsort(np.asarray(pd.DatetimeIndex(dates)))
    d = pd.DatetimeIndex(dates)[order]
    v = np.asarray(vals, float)[order]
    out, starts, cur, last = [], [], [], None
    for di, vi in zip(d, v):
        if last is None or (di - last).days > gap_days:
            if cur:
                out.append(np.mean(cur))
            cur = [vi]
            starts.append(di)
        else:
            cur.append(vi)
        last = di
    if cur:
        out.append(np.mean(cur))
    return np.asarray(out), starts


def rec(v) -> str:
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    w = int((v > 0).sum())
    return f"{w}-{len(v)-w} sign p {sign_test(w, len(v)):.4f}" if len(v) else "0-0"


def dial_ma10() -> pd.Series:
    d = pd.read_parquet(Path(__file__).resolve().parents[3] / "data" / "rd2_fragility.parquet")
    return d["63d"].rolling(10).mean()
