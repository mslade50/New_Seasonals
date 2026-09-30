"""kB shared panel for c3 / c9 / c4 (2026-09-21).

Master calendar = SPY's sessions. Every stock is reindexed to it (US listings
share the NYSE calendar; pre-inception/post-delisting rows are NaN and drop out
of any window that touches them). State definitions match build_pitch_state:
close-based 252-session min for the 52w low, n-day return ranked over a
trailing 252 window for r5/r21/r63. Beta = 252-session OLS slope of daily
returns on SPY, known at the signal close.

Universes:
  LIQ   = strategy_config.LIQUID_PLUS_COMMODITIES single stocks (primary)
  BROAD = every earnings-calendar ticker with prices (robustness row only)
Earnings calendar restricted to 1996+ (registry: fiscal period ends pre-1993).
No BMO/AMC column exists; the pre-print exit is the close of the session BEFORE
the announcement date (p_T - 1), which never holds through a print whether the
name reports before the open or after the close.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / "_kB_panel.pkl"

ETF_LIKE = set("""CEF DIA GLD IBB IHI ITA ITB IWM IYR KRE OIH QQQ SLV SMH SPY UNG
USO UVXY VNQ XBI XHB XLB XLE XLF XLI XLK XLP XLU XLV XLY XME XOP XRT DBC DBA
EEM EFA EWJ EWZ FXI GDX HYG IEF LQD SVXY TLT UUP XLC XLRE""".split())


def _liquid_singles() -> list[str]:
    import strategy_config as sc
    return sorted(t for t in sc.LIQUID_PLUS_COMMODITIES
                  if t not in ETF_LIKE and not t.startswith("^"))


LIQ = _liquid_singles()


def load_earn() -> pd.DataFrame:
    e = pd.read_parquet(ROOT / "data" / "earnings_calendar.parquet",
                        columns=["ticker", "date"])
    e["date"] = pd.to_datetime(e["date"])
    e = e[(e["date"] >= "1996-01-01")]
    e = e[~e.ticker.isin(ETF_LIKE)]
    return e.drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"])


def build_panel():
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    E = load_earn()
    keep = set(E.ticker) | set(LIQ) | {"SPY"}
    mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                         columns=["ticker", "date", "Close"])
    mp = mp[mp.ticker.isin(keep)]
    mp["date"] = pd.to_datetime(mp["date"])
    mp = mp.drop_duplicates(["ticker", "date"], keep="last")
    C = mp.pivot(index="date", columns="ticker", values="Close").sort_index()
    idx = C["SPY"].dropna().index
    C = C.reindex(idx)
    C = C.loc[:, C.notna().sum() >= 300]
    ret = C.pct_change(fill_method=None)
    spy = ret["SPY"]
    # 252d beta, known at the close
    mx = ret.rolling(252, min_periods=200).mean()
    my = spy.rolling(252, min_periods=200).mean()
    mxy = ret.mul(spy, axis=0).rolling(252, min_periods=200).mean()
    vy = spy.rolling(252, min_periods=200).var(ddof=0)
    beta = (mxy - mx.mul(my, axis=0)).div(vy, axis=0)
    lo = C.rolling(252, min_periods=252).min()
    lodist = C / lo - 1.0
    ranks = {}
    for w in (5, 21, 63):
        r = C / C.shift(w) - 1.0
        ranks[w] = r.rolling(252, min_periods=252).rank(pct=True) * 100.0
    sma200 = C["SPY"].rolling(200).mean()
    spy_above200 = C["SPY"] > sma200
    out = dict(C=C, beta=beta, lodist=lodist, r5=ranks[5], r21=ranks[21],
               r63=ranks[63], spy_above200=spy_above200, E=E)
    pd.to_pickle(out, CACHE)
    return out


def event_positions(E: pd.DataFrame, idx: pd.DatetimeIndex, cols) -> pd.DataFrame:
    """p_T = position of the first session on/after the announcement date.
    Drops events outside the price index (no future anchors minted)."""
    E = E[E.ticker.isin(cols)].copy()
    E = E[(E.date >= idx[0]) & (E.date <= idx[-1])]
    E["pT"] = idx.searchsorted(E.date.values)
    E = E[E.pT < len(idx)]
    return E.reset_index(drop=True)


def print_mask(E: pd.DataFrame, idx: pd.DatetimeIndex, cols) -> pd.DataFrame:
    """Boolean panel: True on sessions that are an announcement session p_T."""
    m = pd.DataFrame(False, index=idx, columns=list(cols))
    for t, g in E.groupby("ticker"):
        if t in m.columns:
            m.iloc[g.pT.values, m.columns.get_loc(t)] = True
    return m


def week_cluster(dates: pd.Series, vals: pd.Series) -> pd.Series:
    """Cross-sectional mean per ISO week of the ENTRY date (date clustering:
    many names share a calendar)."""
    d = pd.DatetimeIndex(dates)
    wk = d.to_period("W-FRI")
    return pd.Series(np.asarray(vals, float), index=wk).groupby(level=0).mean()


def stat(vals, label, dates=None) -> dict:
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    r = summarize(v, label)
    if r["n"]:
        w = int((v > 0).sum())
        r["rec"] = f"{w}-{len(v) - w}"
        r["sign_p"] = round(sign_test(w, len(v)), 4)
        if len(v) >= 3:
            r["boot_p"] = round(bootstrap_p_le0(v), 3)
    return r


def cl_stat(df: pd.DataFrame, col: str, label: str) -> dict:
    """Week-clustered summary of column `col` (entry dates in df.entry_date)."""
    if len(df) == 0:
        return {"label": label, "n": 0}
    s = week_cluster(df.entry_date, df[col])
    r = stat(s.values, label)
    r["n_obs"] = len(df)
    return r


def welch(a, b) -> float:
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    b = np.asarray(b, float); b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se
