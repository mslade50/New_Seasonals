"""Shared loaders + calendar features for the seasonality_flow_* studies (2026-09-02).

Basis: flat $750k ledger (data/backtest_trades_full.parquet), per-trade daily MTM
(dist/data/trade_mtm.json, basis flat_750k), per-strategy daily MTM
(dist/data/strategy_daily.json). Trading calendar = SPY sessions in master_prices.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
NAV = 750_000.0
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri"]


def load_ledger() -> pd.DataFrame:
    led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
    led = led[led["PnL_flat_750k"].notna()].copy()
    led["R"] = led["R_Multiple"].astype(float)
    led["pnl"] = led["PnL_flat_750k"].astype(float)
    led["risk"] = led["Risk_flat_750k"].astype(float)
    led["sig"] = pd.to_datetime(led["Signal Date"]).dt.normalize()
    led["yr"] = led["sig"].dt.year
    return led


def load_strategy_daily() -> tuple[pd.DataFrame, pd.Series]:
    sd = json.load(open(ROOT / "dist/data/strategy_daily.json"))
    dates = pd.to_datetime(sd["dates"])
    S = pd.DataFrame(sd["series"], index=dates).fillna(0.0)
    strat = S.T.groupby(S.columns.str.split("||", regex=False).str[0]).sum().T
    tot = pd.Series(sd["total_flat"], index=dates, dtype=float)
    return strat, tot


def load_trade_mtm() -> tuple[pd.DatetimeIndex, dict]:
    t = json.load(open(ROOT / "dist/data/trade_mtm.json"))
    dates = pd.to_datetime(t["dates"])
    m = t["main"]
    return dates, {tid: (s, np.asarray(p, dtype=float)) for tid, s, p in zip(m["trade_id"], m["start"], m["pnl"])}


def load_spy() -> pd.DataFrame:
    px = pq.read_table(ROOT / "data/master_prices.parquet", columns=["ticker", "date", "Close"],
                       filters=[("ticker", "in", ["SPY"])]).to_pandas().set_index("date").sort_index()
    return px


def load_dial() -> pd.Series:
    frag = pd.read_parquet(ROOT / "data/rd2_fragility.parquet")
    return frag["63d"].rolling(10).mean().rename("dial")


def trading_calendar(spy_dates: pd.DatetimeIndex, earnings: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per SPY session with every calendar feature the studies condition on."""
    d = pd.DatetimeIndex(sorted(pd.to_datetime(spy_dates).normalize().unique()))
    cal = pd.DataFrame(index=d)
    cal["month"] = d.month
    cal["mname"] = [MONTHS[m - 1] for m in d.month]
    cal["quarter"] = d.quarter
    cal["year"] = d.year
    cal["dow"] = d.dayofweek
    cal["dname"] = [DOW[x] if x < 5 else "Wknd" for x in d.dayofweek]
    ym = d.to_period("M")
    cal["td_in_month"] = cal.groupby(ym).cumcount() + 1
    cal["td_left_in_month"] = cal.groupby(ym).cumcount(ascending=False) + 1
    # turn of month: last 2 sessions + first 3 sessions (standard -1..+3 window widened by one)
    cal["tom"] = (cal["td_left_in_month"] <= 2) | (cal["td_in_month"] <= 3)
    # opex week: Mon..Fri of the week containing the 3rd Friday
    third_fri = {}
    for p in ym.unique():
        days = pd.date_range(p.start_time, p.end_time, freq="D")
        fr = [x for x in days if x.dayofweek == 4]
        third_fri[p] = fr[2]
    tf = pd.Series([third_fri[p] for p in ym], index=d)
    wk_start = tf - pd.to_timedelta(tf.dt.dayofweek, unit="D")
    cal["opex_week"] = (d >= wk_start.values) & (d <= tf.values)
    cal["opex_day"] = d == tf.values
    cal["post_opex_week"] = (d > tf.values) & (d <= (tf + pd.Timedelta(days=7)).values)
    # week-of-month bucket by session index (W1: 1-5, W2: 6-10, W3: 11-15, W4: 16+)
    cal["wom"] = np.where(cal["td_in_month"] <= 5, "W1", np.where(cal["td_in_month"] <= 10, "W2",
                          np.where(cal["td_in_month"] <= 15, "W3", "W4+")))
    # holiday adjacency: a weekday gap between consecutive sessions marks a market holiday
    nxt = pd.Series(d[1:].append(pd.DatetimeIndex([pd.NaT])), index=d)
    prv = pd.Series(pd.DatetimeIndex([pd.NaT]).append(d[:-1]), index=d)
    bd_to_next = [np.busday_count(a.date(), b.date()) if pd.notna(b) else 1 for a, b in zip(d, nxt)]
    bd_from_prev = [np.busday_count(b.date(), a.date()) if pd.notna(b) else 1 for a, b in zip(d, prv)]
    cal["pre_holiday"] = np.array(bd_to_next) > 1
    cal["post_holiday"] = np.array(bd_from_prev) > 1
    cal["holiday_adj"] = np.where(cal["pre_holiday"], "pre", np.where(cal["post_holiday"], "post", "none"))
    # fixed base-rate earnings season: sessions 8..35 calendar days after quarter end, i.e. ~Jan 12-Feb 20 etc.
    doy_in_q = (d - pd.DatetimeIndex([pd.Timestamp(y, 3 * (q - 1) + 1, 1) for y, q in zip(d.year, d.quarter)])).days
    cal["eseason_fixed"] = (doy_in_q >= 11) & (doy_in_q <= 50)
    # data-driven earnings season: count of reports within +-5 sessions, top tercile within year
    if earnings is not None:
        e = pd.to_datetime(earnings["date"]).dt.normalize()
        e = e[(e >= d[0]) & (e <= d[-1])]
        cnt = e.value_counts().reindex(d).fillna(0.0)
        win = cnt.rolling(11, center=True, min_periods=1).sum()
        thr = win.groupby(d.year).transform(lambda s: s.quantile(0.67))
        cal["ecount11"] = win.values
        cal["eseason_data"] = (win > thr).values
    else:
        cal["ecount11"] = np.nan
        cal["eseason_data"] = cal["eseason_fixed"]
    cal["half"] = np.where(cal["month"].isin([11, 12, 1, 2, 3, 4]), "NovApr", "MayOct")
    return cal


# ------------------------------------------------------------------ stats helpers
def episodes(sig_dates: pd.Series, gap_td: int = 5, cal_index: pd.DatetimeIndex | None = None) -> np.ndarray:
    """Cluster ids for signal dates separated by < gap_td sessions (per strategy)."""
    if cal_index is None:
        pos = sig_dates.values.astype("datetime64[D]").astype(np.int64)  # calendar days fallback
    else:
        pos = cal_index.get_indexer(sig_dates.values)
    order = np.argsort(pos, kind="stable")
    ids = np.zeros(len(pos), dtype=np.int64)
    cur, last = 0, None
    for i in order:
        if last is not None and pos[i] - last >= gap_td:
            cur += 1
        ids[i] = cur
        last = pos[i]
    return ids


def cluster_diff_t(x: np.ndarray, in_cell: np.ndarray, clusters: np.ndarray) -> tuple[float, float, int]:
    """Cluster-robust t for mean(x|in_cell) - mean(x|~in_cell) via OLS on a cell dummy.
    Returns (t, p, n_clusters_with_cell)."""
    x = np.asarray(x, float)
    dmy = in_cell.astype(float)
    n = len(x)
    if in_cell.sum() < 2 or (~in_cell).sum() < 2:
        return np.nan, np.nan, 0
    X = np.column_stack([np.ones(n), dmy])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ x
    resid = x - X @ beta
    meat = np.zeros((2, 2))
    G = 0
    for c in np.unique(clusters):
        m = clusters == c
        u = X[m].T @ resid[m]
        meat += np.outer(u, u)
        G += 1
    V = XtX_inv @ meat @ XtX_inv * (G / max(G - 1, 1))
    se = np.sqrt(max(V[1, 1], 1e-18))
    t = beta[1] / se
    gc = len(np.unique(clusters[in_cell]))
    df = max(min(G, gc) - 1, 1)
    p = 2 * stats.t.sf(abs(t), df)
    return float(t), float(p), int(gc)


def year_paired_t(df: pd.DataFrame, val: str, in_cell: np.ndarray, year: np.ndarray) -> tuple[float, float, int]:
    """Cell-vs-complement mean difference, one observation per year (paired), t over years."""
    y = pd.DataFrame({"v": df[val].values, "c": in_cell, "y": year})
    g = y.groupby(["y", "c"])["v"].mean().unstack()
    if g.shape[1] < 2:
        return np.nan, np.nan, 0
    g = g.dropna()
    diff = g[True] - g[False]
    n = len(diff)
    if n < 3:
        return np.nan, np.nan, n
    t = diff.mean() / (diff.std(ddof=1) / np.sqrt(n)) if diff.std(ddof=1) > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), n - 1) if np.isfinite(t) else np.nan
    return float(t), float(p), int(n)


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qq = np.empty(n)
    qq[order] = np.clip(ranked, 0, 1)
    q[ok] = qq
    return q


def summarize(df: pd.DataFrame) -> dict:
    r = df["R"].values
    out = dict(N=int(len(df)), avgR=float(np.mean(r)) if len(r) else np.nan,
               sdR=float(np.std(r, ddof=1)) if len(r) > 1 else np.nan,
               win=float(np.mean(r > 0)) if len(r) else np.nan,
               sum_pnl=float(df["pnl"].sum()), sum_risk=float(df["risk"].sum()))
    out["R_per_risk"] = out["sum_pnl"] / out["sum_risk"] if out["sum_risk"] > 0 else np.nan
    out["sharpe_R"] = out["avgR"] / out["sdR"] if out.get("sdR") and out["sdR"] > 0 else np.nan
    return out


def maxdd(pnl: pd.Series) -> float:
    c = pnl.cumsum()
    return float((c - c.cummax()).min())


def perf(pnl: pd.Series) -> dict:
    r = pnl / NAV
    ann = r.mean() * 252
    vol = r.std() * np.sqrt(252)
    dd = maxdd(pnl) / NAV
    return dict(total_pnl=float(pnl.sum()), ann_ret_pct=float(ann * 100), ann_vol_pct=float(vol * 100),
                sharpe=float(ann / vol) if vol > 0 else np.nan, maxdd_pct=float(dd * 100),
                pnl_over_maxdd=float(ann / abs(dd)) if dd < 0 else np.nan,
                worst_day_pct=float(r.min() * 100))


def jdump(obj, path: Path):
    def conv(o):
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if isinstance(o, (pd.Timestamp,)):
            return str(o.date())
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple, np.ndarray)):
            return [conv(v) for v in o]
        return o
    path.write_text(json.dumps(conv(obj), indent=1))
