"""Risk Agent research helpers for the agent's check scripts.

Import from scratch/risk_agent_checks/<asof>/*.py:

    import sys; sys.path.insert(0, r"<repo root>")
    import risk_agent_lab as lab
    px = lab.prices(["SPY", "TLT"])
    fwd = lab.fwd_returns(px["SPY"], 21)          # FRACTIONS
    out = lab.study(mask, fwd, decluster_td=10)   # PERCENT in the summary

Conventions (same as pitch_lab):
  * Inputs and series are FRACTIONS (0.012 = 1.2%).
  * Summary dicts (study) report PERCENT (mean_pct, median_pct, hit in %,
    worst_pct, best_pct). `t` is a plain t-stat, `sign_p` a one-sided exact
    binomial p from pitch_lab.sign_test.
  * Closes are RAW (auto_adjust=False style) from the Risk Agent cache, the
    same basis the paper ledger fills and marks on.

Reads only risk_agent_data's cache (allowlisted R2 objects). Agent-product
module: the book must not import it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import risk_agent_data as rad
from pitch_lab import sign_test

_PX_CACHE: dict = {}


def _master(cache_dir=None, symbols=None) -> pd.DataFrame:
    cdir = Path(cache_dir or rad.CACHE_DIR)
    p = rad.local_path("master_prices.parquet", cdir)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run risk_agent_data.sync()")
    filters = [("ticker", "in", list(symbols))] if symbols is not None else None
    df = pd.read_parquet(p, filters=filters)
    df["date"] = pd.to_datetime(df["date"])
    return df


def prices(symbols=None, field: str = "Close", start=None, cache_dir=None) -> pd.DataFrame:
    """Wide DataFrame (date x ticker) of RAW `field`. symbols=None loads all (slow)."""
    syms = None if symbols is None else sorted(set(symbols))
    key = (str(cache_dir), tuple(syms) if syms else None, field, str(start))
    if key in _PX_CACHE:
        return _PX_CACHE[key].copy()
    df = _master(cache_dir, syms)
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    wide = df.pivot_table(index="date", columns="ticker", values=field, aggfunc="last")
    wide = wide.astype("float64").sort_index()
    _PX_CACHE[key] = wide
    return wide.copy()


def ohlc(symbol: str, cache_dir=None) -> pd.DataFrame:
    """Open/High/Low/Close/Volume for one symbol, date index, raw, float64."""
    df = _master(cache_dir, [symbol])
    df = df.sort_values("date").set_index("date")[["Open", "High", "Low", "Close", "Volume"]]
    return df.astype("float64")


def fwd_returns(close, h: int, lag: int = 1):
    """Forward h-session return as a FRACTION, aligned to the SIGNAL date.

    lag=1 (default): signal is known at the close of D, you enter at the close
    of D+1 (MOC tomorrow; a proxy for MOO at the D+1 open) and exit h sessions
    later: close[D+1+h] / close[D+1] - 1. lag=0 enters at the signal close
    (look-ahead for any signal computed from that close; use only for
    unconditional baselines). Same convention as pitch_lab.fwd_lag.
    """
    return close.shift(-(lag + h)) / close.shift(-lag) - 1.0


def _decluster(idx: pd.DatetimeIndex, all_dates: pd.DatetimeIndex, gap: int) -> list:
    pos = pd.Series(np.arange(len(all_dates)), index=all_dates)
    keep, last = [], -10 ** 9
    for d in sorted(idx):
        p = pos.get(d)
        if p is None:
            continue
        if p - last >= gap:
            keep.append(d)
            last = p
    return keep


def _stats(v: np.ndarray) -> dict:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return {"n": 0, "mean_pct": None, "median_pct": None, "hit": None, "t": None,
                "worst_pct": None, "best_pct": None}
    sd = v.std(ddof=1) if n > 1 else np.nan
    t = v.mean() / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan
    return {"n": n, "mean_pct": 100 * float(v.mean()), "median_pct": 100 * float(np.median(v)),
            "hit": 100 * float((v > 0).mean()), "t": float(t),
            "worst_pct": 100 * float(v.min()), "best_pct": 100 * float(v.max())}


def study(mask, fwd, decluster_td: int | None = None) -> dict:
    """Conditional vs unconditional forward returns.

    mask: boolean Series (signal dates). fwd: forward-return Series (FRACTIONS,
    e.g. fwd_returns). decluster_td keeps the first signal of each cluster
    (gap in sessions of the fwd index). Returns percent figures:
      n, mean_pct, median_pct, hit (% positive), t, sign_p (one-sided exact
      binomial of wins vs the UNCONDITIONAL hit rate), worst_pct, best_pct,
      plus `uncond` (same stats over every date with a defined fwd) and
      `edge_mean_pct` (conditional minus unconditional mean).
    """
    fwd = pd.Series(fwd).dropna()
    m = pd.Series(mask).reindex(fwd.index).fillna(False).astype(bool)
    idx = fwd.index[m.to_numpy()]
    if decluster_td:
        idx = pd.DatetimeIndex(_decluster(idx, fwd.index, int(decluster_td)))
    cond = fwd.loc[idx].to_numpy()
    uncond = _stats(fwd.to_numpy())
    out = _stats(cond)
    wins = int((cond > 0).sum())
    base_p = uncond["hit"] / 100.0 if uncond["hit"] is not None else 0.5
    out["sign_p"] = float(sign_test(wins, len(cond), base_p)) if len(cond) else None
    out["uncond"] = uncond
    out["edge_mean_pct"] = (out["mean_pct"] - uncond["mean_pct"]) if out["n"] and uncond["n"] else None
    return out


def _dashboard_json(cache_dir=None) -> dict:
    p = rad.local_path("shared/site_risk.json", cache_dir or rad.CACHE_DIR)
    return json.loads(p.read_text(encoding="utf-8"))


def dashboard_history(cache_dir=None) -> pd.DataFrame:
    """Each dashboard signal's metric over time (date index, one column per
    signal named by the signal, plus `dial_5d`, `dial_21d`, `dial_63d`
    fragility series when present). Values are in each signal's own unit."""
    d = _dashboard_json(cache_dir)
    dates = pd.to_datetime(d.get("dates") or [])
    cols = {}
    for name, det in (d.get("signal_detail") or {}).items():
        vals = (det.get("metric") or {}).get("values") or []
        if len(vals) == len(dates):
            cols[name] = pd.Series(vals, index=dates, dtype="float64")
    fs = d.get("fragility_series") or {}
    for k, vals in fs.items():
        if len(vals) == len(dates):
            cols[f"dial_{k}"] = pd.Series(vals, index=dates, dtype="float64")
    return pd.DataFrame(cols)


def chain(underlying: str, date=None, cache_dir=None) -> pd.DataFrame:
    """Option chain rows for one underlying on a snapshot date (default latest).
    Columns as in options/positioning_history plus `expiry_date` (Timestamp)."""
    p = rad.local_path("options/positioning_history.parquet", cache_dir or rad.CACHE_DIR)
    df = pd.read_parquet(p, filters=[("ticker", "==", underlying)])
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    d = pd.Timestamp(date) if date is not None else df["date"].max()
    out = df[df["date"] == d].copy()
    out["expiry_date"] = pd.to_datetime(out["expiry"].astype(str), format="%Y%m%d", errors="coerce")
    return out.sort_values(["expiry_date", "strike", "right"]).reset_index(drop=True)


def iv_history(ticker: str, cache_dir=None) -> pd.Series:
    """30-day IV history (decimal, 0.20 = 20 vol) indexed by date."""
    p = rad.local_path("options/iv_history.parquet", cache_dir or rad.CACHE_DIR)
    df = pd.read_parquet(p, filters=[("ticker", "==", ticker)])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").set_index("date")["iv30"].rename(ticker)
