"""Shared reference-class machinery for the 2026-08-28 stage-C 'b' checks.

Not a re-implementation of pitch_lab -- it CALLS pitch_lab for every statistic
and only adds the pooled/heterogeneity layer the reference-class kill needs
(Cochran Q, I-squared, fixed-effect common excess, permutation max-of-N).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import (declusters, load_prices, local_control, pct_rank,
                       summarize)


def series(px: dict, t: str) -> pd.Series:
    return px[t]["Close"].dropna()


def fwd(s: pd.Series, h: int, lag: int = 1) -> pd.Series:
    return s.shift(-(lag + h)) / s.shift(-lag) - 1.0


def per_name(px: dict, tickers: list[str], mask_fn, h: int, min_gap: int,
             lag: int = 1) -> pd.DataFrame:
    """One row per ticker: conditional episode stats, own-drift control and
    the excess. mask_fn(series) -> boolean Series on that ticker's own index."""
    rows = []
    for t in tickers:
        s = series(px, t)
        r = fwd(s, h, lag)
        valid = r.dropna().index
        m = mask_fn(s).reindex(s.index, fill_value=False)
        trig = s.index[m.values].intersection(valid)
        if len(trig) == 0:
            rows.append({"tkr": t, "n_days": 0, "n_epi": 0})
            continue
        epi = declusters(trig, min_gap, valid)
        v = r.loc[epi].values
        v = v[~np.isnan(v)]
        span = (valid >= trig[0]) & (valid <= trig[-1])
        ctrl = r.loc[valid[span]].values
        ctrl = ctrl[~np.isnan(ctrl)]
        loc = local_control(valid, trig)
        lv = r.loc[loc].values
        lv = lv[~np.isnan(lv)]
        sd = v.std(ddof=1) if len(v) > 1 else np.nan
        se = sd / np.sqrt(len(v)) if len(v) > 1 else np.nan
        exc = v.mean() - ctrl.mean() if len(ctrl) else np.nan
        se_d = (np.sqrt(v.var(ddof=1) / len(v) + ctrl.var(ddof=1) / len(ctrl))
                if len(v) > 1 and len(ctrl) > 1 else np.nan)
        rows.append({
            "tkr": t, "n_days": len(trig), "n_epi": len(v),
            "mean_pct": 100 * v.mean(),
            "hit": 100 * (v > 0).mean(),
            "drift_pct": 100 * ctrl.mean() if len(ctrl) else np.nan,
            "local_pct": 100 * lv.mean() if len(lv) else np.nan,
            "excess_pct": 100 * exc,
            "t_excess": exc / se_d if se_d and se_d > 0 else np.nan,
            "se_pct": 100 * se,
            "worst_pct": 100 * v.min(),
            "first": str(trig[0].date()), "last": str(trig[-1].date()),
        })
    return pd.DataFrame(rows)


def cochran(df: pd.DataFrame, val="excess_pct", se="se_d_pct") -> dict:
    """Fixed-effect meta over the per-name excesses. Q, df, p, I-squared."""
    from scipy import stats
    d = df.dropna(subset=[val, se])
    d = d[d[se] > 0]
    if len(d) < 2:
        return {}
    w = 1.0 / d[se].values ** 2
    y = d[val].values
    mu = (w * y).sum() / w.sum()
    Q = float((w * (y - mu) ** 2).sum())
    k = len(d)
    p = float(1 - stats.chi2.cdf(Q, k - 1))
    I2 = max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0
    se_mu = float(np.sqrt(1.0 / w.sum()))
    return {"k": k, "Q": Q, "df": k - 1, "p": p, "I2_pct": I2,
            "fe_common_pct": float(mu), "fe_se_pct": se_mu,
            "fe_t": float(mu / se_mu)}


def perm_max_of_n(px: dict, tickers: list[str], mask_fn, h: int, min_gap: int,
                  n_perm: int = 400, seed: int = 42, lag: int = 1) -> dict:
    """Permutation max-of-N null: for each name, keep the trigger COUNT and
    the episode spacing but shift the trigger block by a random circular
    offset in that name's own return series. Records the max |t| and max
    excess across the family per draw -> family-wise p for the observed best.
    """
    rng = np.random.default_rng(seed)
    prepped = []
    for t in tickers:
        s = series(px, t)
        r = fwd(s, h, lag)
        valid = r.dropna().index
        m = mask_fn(s).reindex(s.index, fill_value=False)
        trig = s.index[m.values].intersection(valid)
        if len(trig) < 3:
            continue
        epi = declusters(trig, min_gap, valid)
        pos = pd.Series(range(len(valid)), index=valid)
        ip = np.array([pos[d] for d in epi if d in pos.index])
        prepped.append((t, r.loc[valid].values, ip))
    obs_max_exc, obs_max_t, obs_name = -1e9, -1e9, None
    for t, rv, ip in prepped:
        v = rv[ip]
        exc = v.mean() - rv.mean()
        tt = exc / (v.std(ddof=1) / np.sqrt(len(v)))
        if exc > obs_max_exc:
            obs_max_exc, obs_name = exc, t
        obs_max_t = max(obs_max_t, tt)
    null_exc, null_t = [], []
    for _ in range(n_perm):
        me, mt = -1e9, -1e9
        for t, rv, ip in prepped:
            off = rng.integers(0, len(rv))
            idx = (ip + off) % len(rv)
            v = rv[idx]
            exc = v.mean() - rv.mean()
            tt = exc / (v.std(ddof=1) / np.sqrt(len(v)))
            me, mt = max(me, exc), max(mt, tt)
        null_exc.append(me)
        null_t.append(mt)
    return {
        "best_name": obs_name,
        "obs_max_excess_pct": 100 * obs_max_exc,
        "obs_max_t": obs_max_t,
        "fw_p_excess": float((np.array(null_exc) >= obs_max_exc).mean()),
        "fw_p_t": float((np.array(null_t) >= obs_max_t).mean()),
        "null_excess_p95_pct": 100 * float(np.quantile(null_exc, 0.95)),
        "n_names": len(prepped), "n_perm": n_perm,
    }


def pooled(px: dict, tickers: list[str], mask_fn, h: int, min_gap: int,
           label: str, lag: int = 1) -> dict:
    """Pool every name's declustered episodes into one cell (name-agnostic)."""
    vals, dates, names = [], [], []
    for t in tickers:
        s = series(px, t)
        r = fwd(s, h, lag)
        valid = r.dropna().index
        m = mask_fn(s).reindex(s.index, fill_value=False)
        trig = s.index[m.values].intersection(valid)
        if len(trig) == 0:
            continue
        epi = declusters(trig, min_gap, valid)
        v = r.loc[epi].values
        ok = ~np.isnan(v)
        vals.extend(v[ok])
        dates.extend(np.asarray(epi)[ok])
        names.extend([t] * int(ok.sum()))
    out = summarize(np.asarray(vals), label)
    out["_vals"] = np.asarray(vals)
    out["_dates"] = pd.DatetimeIndex(dates)
    out["_names"] = np.asarray(names)
    return out
