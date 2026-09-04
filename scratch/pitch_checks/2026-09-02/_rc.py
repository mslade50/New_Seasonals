"""Reference-class machinery for the 2026-09-02 stage-C checks.

Same contract as scratch/pitch_checks/2026-08-28/b0_pool.py -- it CALLS
pitch_lab for every statistic and only adds the pooled/heterogeneity layer
(Cochran Q, I-squared, fixed-effect common excess, permutation max-of-N).
Re-stated in today's folder because the survey rule wants every evidence
script resolvable inside scratch/pitch_checks/2026-09-02/.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402

from pitch_lab import declusters, local_control, summarize  # noqa: E402


def fwd(s: pd.Series, h: int, lag: int = 1) -> pd.Series:
    return s.shift(-(lag + h)) / s.shift(-lag) - 1.0


def per_name(px: dict, tickers: list[str], mask_fn, h: int, min_gap: int,
             lag: int = 1) -> pd.DataFrame:
    rows = []
    for t in tickers:
        s = px[t].dropna()
        r = fwd(s, h, lag)
        valid = r.dropna().index
        m = mask_fn(t, s).reindex(s.index, fill_value=False)
        trig = s.index[m.values].intersection(valid)
        if len(trig) == 0:
            rows.append({"tkr": t, "n_days": 0, "n_epi": 0})
            continue
        epi = declusters(trig, min_gap, valid)
        v = r.loc[epi].values
        v = v[~np.isnan(v)]
        if len(v) < 2:
            rows.append({"tkr": t, "n_days": len(trig), "n_epi": len(v)})
            continue
        span = (valid >= trig[0]) & (valid <= trig[-1])
        ctrl = r.loc[valid[span]].values
        ctrl = ctrl[~np.isnan(ctrl)]
        loc = local_control(valid, trig)
        lv = r.loc[loc].values
        lv = lv[~np.isnan(lv)]
        exc = v.mean() - ctrl.mean()
        se_d = np.sqrt(v.var(ddof=1) / len(v) + ctrl.var(ddof=1) / len(ctrl))
        rows.append({
            "tkr": t, "n_days": len(trig), "n_epi": len(v),
            "mean_pct": 100 * v.mean(),
            "hit": 100 * (v > 0).mean(),
            "drift_pct": 100 * ctrl.mean(),
            "local_pct": 100 * lv.mean() if len(lv) else np.nan,
            "excess_pct": 100 * exc,
            "t_excess": exc / se_d if se_d > 0 else np.nan,
            "se_d_pct": 100 * se_d,
            "worst_pct": 100 * v.min(),
            "first": str(trig[0].date()), "last": str(trig[-1].date()),
        })
    return pd.DataFrame(rows)


def cochran(df: pd.DataFrame, val="excess_pct", se="se_d_pct") -> dict:
    d = df.dropna(subset=[val, se])
    d = d[d[se] > 0]
    if len(d) < 2:
        return {}
    w = 1.0 / d[se].values ** 2
    y = d[val].values
    mu = (w * y).sum() / w.sum()
    Q = float((w * (y - mu) ** 2).sum())
    k = len(d)
    p = float(1 - sps.chi2.cdf(Q, k - 1))
    I2 = max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0
    se_mu = float(np.sqrt(1.0 / w.sum()))
    return {"k": k, "Q": Q, "df": k - 1, "p": p, "I2_pct": I2,
            "fe_common_pct": float(mu), "fe_se_pct": se_mu,
            "fe_t": float(mu / se_mu)}


def perm_max_of_n(px: dict, tickers: list[str], mask_fn, h: int, min_gap: int,
                  n_perm: int = 1000, seed: int = 42, lag: int = 1) -> dict:
    """Correlation-preserving permutation: shift each name's trigger BLOCK by
    ONE COMMON random circular offset per draw, so the cross-name correlation
    that makes nine SPDRs move together is retained in the null."""
    rng = np.random.default_rng(seed)
    prepped = []
    for t in tickers:
        s = px[t].dropna()
        r = fwd(s, h, lag)
        valid = r.dropna().index
        m = mask_fn(t, s).reindex(s.index, fill_value=False)
        trig = s.index[m.values].intersection(valid)
        if len(trig) < 3:
            continue
        epi = declusters(trig, min_gap, valid)
        pos = pd.Series(range(len(valid)), index=valid)
        ip = np.array([pos[d] for d in epi if d in pos.index])
        prepped.append((t, r.loc[valid].values, ip, valid))
    obs = {}
    for t, rv, ip, _ in prepped:
        v = rv[ip]
        exc = v.mean() - rv.mean()
        tt = exc / (v.std(ddof=1) / np.sqrt(len(v)))
        obs[t] = (exc, tt)
    null_exc, null_t = [], []
    # a common offset in CALENDAR position keeps the family aligned
    max_len = min(len(rv) for _, rv, _, _ in prepped)
    for _ in range(n_perm):
        off = int(rng.integers(0, max_len))
        me, mt = -1e9, -1e9
        for t, rv, ip, _ in prepped:
            idx = (ip + off) % len(rv)
            v = rv[idx]
            sd = v.std(ddof=1)
            exc = v.mean() - rv.mean()
            tt = exc / (sd / np.sqrt(len(v))) if sd > 0 else 0.0
            me, mt = max(me, exc), max(mt, abs(tt))
        null_exc.append(me)
        null_t.append(mt)
    return {"obs": obs, "null_exc": np.array(null_exc),
            "null_t": np.array(null_t), "n_names": len(prepped),
            "n_perm": n_perm}


def pooled(px: dict, tickers: list[str], mask_fn, h: int, min_gap: int,
           label: str, lag: int = 1) -> dict:
    vals, dates, names = [], [], []
    for t in tickers:
        s = px[t].dropna()
        r = fwd(s, h, lag)
        valid = r.dropna().index
        m = mask_fn(t, s).reindex(s.index, fill_value=False)
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


def jaccard(a: pd.DatetimeIndex, b: pd.DatetimeIndex) -> tuple[int, int, float]:
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter, union, (inter / union if union else float("nan"))


def welch(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    return (x.mean() - y.mean()) / se if se > 0 else float("nan")


def dial_series() -> pd.Series:
    """10d MA of the 63d fragility column -- the sizing statistic. Says which
    vintage: rows before 2026-07-02 are the RECOMPUTE vintage, later rows are
    point-in-time appends (CLAUDE.md 'vintage rule')."""
    p = Path(__file__).resolve().parents[3] / "data" / "rd2_fragility.parquet"
    f = pd.read_parquet(p)
    if "date" in f.columns:
        f = f.set_index(pd.to_datetime(f["date"]))
    f.index = pd.DatetimeIndex(f.index).normalize()
    col = "63d" if "63d" in f.columns else [c for c in f.columns if "63" in str(c)][0]
    return f[col].rolling(10).mean()
