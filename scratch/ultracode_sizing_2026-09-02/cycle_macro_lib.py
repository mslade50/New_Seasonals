"""Shared loaders, regime builders and statistics for the cycle/macro regime
conditioning study (2026-09-02). Imported by cycle_macro_01..03.

Conventions
- dollars: flat $750k basis (ledger PnL_flat_750k, strategy_daily.json)
- trade regime: indicator value at the SIGNAL DATE close (known when the PM
  scan sizes the order); P/C fear is lag-1 like the book's pc_fear.py
- daily-series regime: indicator lagged one session (yesterday's close sizes
  today's holdings)
- dial: 10d MA of the 63d column of data/rd2_fragility.parquet; rows before
  2026-07-02 are the RECOMPUTE vintage (stated on every table that uses it)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
NAV = 750_000.0

FAMILY_BAND = ["Weak Close Decent Sznls", "SPY QQQ MonFri Reversion", "Monday Dip",
               "Indices Oversold Bounce", "3x Bear ETF Overbot Fade", "Monthly Weak Close"]
DIAL_GATED = {"52wh Breakout": 30.0, "St OS Sznl": 65.0}
CYCLE_TILTED = {"Overbot Vol Spike": {2: 0.75}}
CYCLE_NAMES = {0: "election", 1: "post_election", 2: "midterm", 3: "pre_election"}


def load_ledger() -> pd.DataFrame:
    led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
    led = led[led["PnL_flat_750k"].notna()].copy()
    led["Strat2"] = led["Strategy"]
    ovs = led["Strategy"] == "Overbot Vol Spike"
    led.loc[ovs & (led["Risk bps"] >= 30), "Strat2"] = "OVS path 1"
    led.loc[ovs & (led["Risk bps"] < 30), "Strat2"] = "OVS path 2"
    led["yr"] = led["Signal Date"].dt.year
    led["ym"] = led["Signal Date"].dt.to_period("M").astype(str)
    return led


def load_daily() -> tuple[pd.DataFrame, pd.Series]:
    sd = json.load(open(ROOT / "dist/data/strategy_daily.json"))
    dates = pd.to_datetime(sd["dates"])
    S = pd.DataFrame(sd["series"], index=dates).fillna(0.0)
    strat = S.T.groupby(S.columns.str.split("||", regex=False).str[0]).sum().T / NAV
    tot = pd.Series(sd["total_flat"], index=dates, dtype=float) / NAV
    return strat, tot


def load_prices() -> pd.DataFrame:
    want = ["SPY", "QQQ", "IWM", "^VIX", "^VIX3M", "^TNX", "HYG", "LQD", "TLT"]
    t = pq.read_table(ROOT / "data/master_prices.parquet", columns=["ticker", "date", "Close"],
                      filters=[("ticker", "in", want)]).to_pandas()
    px = t.pivot(index="date", columns="ticker", values="Close").sort_index()
    px.index = pd.to_datetime(px.index).normalize()
    return px


def pc_fear_pct() -> pd.Series:
    """Trailing-252d percentile of the 10d-MA CBOE equity P/C, indexed by DATA
    date. Mirrors pc_fear.pct_series (no lag applied here)."""
    eq = pd.read_parquet(ROOT / "data/cboe_putcall.parquet")["equity"].dropna().sort_index()
    eq.index = pd.to_datetime(eq.index).normalize()
    ma = eq.rolling(10, min_periods=10).mean()
    return ma.rolling(252, min_periods=252).apply(lambda w: (w <= w[-1]).mean() * 100.0, raw=True).dropna()


def build_regimes() -> pd.DataFrame:
    """One row per SPY session. Every column is a categorical regime label
    computed from data through that session's close (no lag inside)."""
    px = load_prices()
    spy = px["SPY"].dropna()
    idx = spy.index
    R = pd.DataFrame(index=idx)
    R["cycle"] = [CYCLE_NAMES[y % 4] for y in idx.year]
    # cycle half: H1/H2 of the year (midterm lows cluster in Q2-Q3)
    R["cycle_half"] = R["cycle"] + np.where(idx.month <= 6, "_H1", "_H2")
    vix = px["^VIX"].reindex(idx).ffill()
    R["vix_lvl"] = pd.cut(vix, [0, 15, 20, 30, 999], labels=["<15", "15-20", "20-30", "30+"]).astype(str)
    v3 = px["^VIX3M"].reindex(idx)
    ratio = (vix / v3)
    R["vix_ts"] = pd.cut(ratio, [0, 0.9, 1.0, 9], labels=["contango_steep", "contango_mild", "backwardation"]).astype(str)
    R.loc[ratio.isna(), "vix_ts"] = "nan"
    tnx = px["^TNX"].reindex(idx).ffill()
    d63 = tnx - tnx.shift(63)
    R["tnx_chg63"] = pd.cut(d63, [-99, -0.25, 0.25, 99], labels=["falling", "flat", "rising"]).astype(str)
    R["tnx_lvl"] = pd.cut(tnx, [0, 2, 4, 99], labels=["<2", "2-4", ">4"]).astype(str)
    hl = (px["HYG"] / px["LQD"]).reindex(idx)
    hl21 = np.log(hl) - np.log(hl.shift(21))
    R["credit21"] = pd.cut(hl21, [-9, -0.015, 0.015, 9], labels=["widening", "flat", "tightening"]).astype(str)
    R.loc[hl21.isna(), "credit21"] = "nan"
    sma200 = spy.rolling(200).mean()
    R["spy_200"] = np.where(spy > sma200, "above", "below")
    R.loc[sma200.isna(), "spy_200"] = "nan"
    mom = spy.shift(21) / spy.shift(252) - 1
    R["mom12_1"] = np.where(mom > 0, "pos", "neg")
    R.loc[mom.isna(), "mom12_1"] = "nan"
    lr = np.log(spy).diff()
    rv21 = lr.rolling(21).std() * np.sqrt(252) * 100
    R["rv21"] = pd.cut(rv21, [0, 12, 20, 30, 999], labels=["<12", "12-20", "20-30", "30+"]).astype(str)
    R.loc[rv21.isna(), "rv21"] = "nan"
    vr = (lr.rolling(10).std() / lr.rolling(63).std())
    R["vol_ratio"] = pd.cut(vr, [0, 0.8, 1.3, 99], labels=["contracting", "steady", "expanding"]).astype(str)
    R.loc[vr.isna(), "vol_ratio"] = "nan"
    hi252 = spy.rolling(252, min_periods=120).max()
    dd = spy / hi252 - 1
    R["spy_dd"] = pd.cut(dd, [-1, -0.20, -0.10, -0.03, 0.001], labels=["bear>20", "10-20", "3-10", "<3"]).astype(str)
    R.loc[dd.isna(), "spy_dd"] = "nan"
    # P/C fear, lag-1 by construction (row dated <= D-1 bday)
    pc = pc_fear_pct()
    pc_lag = pc.reindex(idx, method="ffill").shift(1)
    R["pc_fear"] = np.where(pc_lag > 85, "fear_on", np.where(pc_lag < 10, "complacent", "mid"))
    R.loc[pc_lag.isna(), "pc_fear"] = "nan"
    # dial (current-weights vintage; recompute before 2026-07-02)
    frag = pd.read_parquet(ROOT / "data/rd2_fragility.parquet")
    dial = frag["63d"].rolling(10).mean()
    dial.index = pd.to_datetime(dial.index).normalize()
    dl = dial.reindex(idx, method="ffill")
    dl[idx < dial.index.min()] = np.nan
    R["dial"] = pd.cut(dl, [0, 30, 50, 65, 101], labels=["<30", "30-50", "50-65", "65+"], right=False).astype(str)
    R.loc[dl.isna(), "dial"] = "nan"
    R["dial_val"] = dl
    R["vix_val"] = vix
    R["spy_ret"] = spy.pct_change()
    R["rv21_val"] = rv21
    R["vix_ts_val"] = ratio
    R["spy_dd_val"] = dd
    R["pc_val"] = pc_lag
    return R


REGIME_COLS = ["cycle", "cycle_half", "vix_lvl", "vix_ts", "tnx_chg63", "tnx_lvl", "credit21", "spy_200", "mom12_1",
               "rv21", "vol_ratio", "spy_dd", "pc_fear", "dial"]


def attach_trade_regimes(led: pd.DataFrame, R: pd.DataFrame) -> pd.DataFrame:
    """Regime at the SIGNAL DATE close (asof merge backward)."""
    cols = REGIME_COLS + ["dial_val", "vix_val", "rv21_val", "vix_ts_val", "spy_dd_val", "pc_val"]
    r = R[cols].copy()
    r.index.name = "date"
    out = pd.merge_asof(led.sort_values("Signal Date"), r.reset_index().rename(columns={"date": "Signal Date"}).sort_values("Signal Date"),
                        on="Signal Date", direction="backward")
    return out


def episode_ids(mask: pd.Series, gap: int = 21) -> pd.Series:
    """Cluster ids for a boolean daily mask: runs of True separated by >= gap
    sessions of False get distinct ids; False days get -1."""
    m = mask.fillna(False).values.astype(bool)
    ids = np.full(len(m), -1)
    cur, last_true = 0, -10**9
    for i, v in enumerate(m):
        if v:
            if i - last_true > gap:
                cur += 1
            ids[i] = cur
            last_true = i
    return pd.Series(ids, index=mask.index)


def cluster_t(y: np.ndarray, x: np.ndarray, clusters: np.ndarray) -> tuple[float, float, int]:
    """OLS y ~ 1 + x with CR1 cluster-robust SE. Returns (beta, t, n_clusters)."""
    X = np.column_stack([np.ones(len(y)), x.astype(float)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    meat = np.zeros((2, 2))
    ug = np.unique(clusters)
    for g in ug:
        i = clusters == g
        s = X[i].T @ e[i]
        meat += np.outer(s, s)
    G, n, k = len(ug), len(y), 2
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k)) if G > 1 else np.full((2, 2), np.nan)
    se = np.sqrt(V[1, 1])
    return float(beta[1]), float(beta[1] / se) if se > 0 else np.nan, int(G)


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    return float((a.mean() - b.mean()) / np.sqrt(va + vb)) if va + vb > 0 else np.nan


def loyo(df: pd.DataFrame, mask: pd.Series, col: str = "R_Multiple") -> dict:
    """Leave-one-year-out stability of avgR(mask) - avgR(~mask).
    Returns min/max of the LOYO effect, and per-year sign agreement of the
    held-out year's own effect."""
    yrs = sorted(df["yr"].unique())
    effs, signs = [], []
    for y in yrs:
        tr = df[df["yr"] != y]; te = df[df["yr"] == y]
        mt, mte = mask[tr.index], mask[te.index]
        if mt.sum() >= 5 and (~mt).sum() >= 5:
            effs.append(tr.loc[mt, col].mean() - tr.loc[~mt, col].mean())
        if mte.sum() >= 3 and (~mte).sum() >= 3:
            signs.append(np.sign(te.loc[mte, col].mean() - te.loc[~mte, col].mean()))
    signs = np.array(signs)
    return dict(loyo_min=float(np.min(effs)) if effs else np.nan, loyo_max=float(np.max(effs)) if effs else np.nan,
                loyo_n=len(effs), yr_pos=int((signs > 0).sum()), yr_neg=int((signs < 0).sum()))


def daily_sharpe(x: pd.Series) -> dict:
    x = x.dropna()
    if len(x) < 15 or x.std() == 0:
        return dict(days=int(len(x)), sharpe=np.nan, mean_bps=np.nan, sd_bps=np.nan)
    return dict(days=int(len(x)), sharpe=float(x.mean() / x.std() * np.sqrt(252)), mean_bps=float(x.mean() * 1e4),
                sd_bps=float(x.std() * 1e4))


def jsonable(o):
    if isinstance(o, (np.floating, float)):
        return None if (isinstance(o, float) and np.isnan(o)) or (isinstance(o, np.floating) and np.isnan(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, pd.Timestamp):
        return o.date().isoformat()
    return o
