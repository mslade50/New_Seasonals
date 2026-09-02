"""Shared helpers for the signal-flow-conditional sizing study (2026-09-02).

Conventions: flat $750k basis; flow counts are INCLUSIVE of the signal day
(the same-day count is known at the close when the order is staged, which is
how same_day_signal_derate already works); trailing windows of k trading days
end on the signal day. Episode = run of family signal dates with gaps <= 5 td.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
OUT_DIR = ROOT / "scratch/ultracode_sizing_2026-09-02"
NAV = 750_000.0
BDAYS = pd.bdate_range("2002-01-01", "2026-12-31")

FAMILY = {
    "Weak Close Decent Sznls": "dip_buy", "SPY QQQ MonFri Reversion": "dip_buy", "Monday Dip": "dip_buy",
    "Indices Oversold Bounce": "dip_buy", "Monthly Weak Close": "dip_buy", "3x Bear ETF Overbot Fade": "dip_buy",
    "Oversold Low Volume": "oversold_hold", "LT Trend ST OS": "oversold_hold", "St OS Sznl": "oversold_hold",
    "Overbot Vol Spike": "short_fade", "ATR Extended Gap Up": "short_fade", "3x ETF Overbot Fade": "short_fade",
    "3x Leader Gap Fade": "short_fade",
    "52wh Breakout": "breakout", "Sector BO": "breakout",
}
FAMILIES = ["dip_buy", "oversold_hold", "short_fade", "breakout"]


def load_ledger() -> pd.DataFrame:
    """Ledger collapsed to TRADES (OVS near/far tranches summed). Adds cap_scale,
    the per-strategy 250 bps daily cap's pro-rata factor recovered from the row:
    Risk_flat = nominal_bps * NAV * Size_Mult * cap_scale."""
    led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
    led = led[led["PnL_flat_750k"].notna()].copy()
    key = ["Strategy", "Tier", "Ticker", "Signal Date", "Entry Date", "Direction"]
    agg = led.groupby(key, as_index=False).agg(
        PnL=("PnL_flat_750k", "sum"), Risk=("Risk_flat_750k", "sum"), ExitDate=("Exit Date", "max"),
        RiskBps=("Risk bps", "first"), SizeMult=("Size_Mult", "first"), ATR=("ATR", "first"),
        EntryPrice=("Entry Price", "first"), Shares=("Shares_flat", "sum"), ExitType=("Exit Type", "first"),
        n_rows=("trade_id", "size"), trade_id=("trade_id", "min"))
    agg["R"] = agg["PnL"] / agg["Risk"]
    nominal = agg["RiskBps"] / 1e4 * NAV * agg["SizeMult"]
    agg["cap_scale"] = (agg["Risk"] / nominal).clip(upper=1.0001)
    agg["family"] = agg["Strategy"].map(FAMILY)
    agg["year"] = agg["Signal Date"].dt.year
    return agg.sort_values(["Signal Date", "Strategy", "Ticker"]).reset_index(drop=True)


def load_candidates() -> pd.DataFrame | None:
    p = OUT_DIR / "flow_candidates.parquet"
    if not p.exists():
        return None
    c = pd.read_parquet(p)
    c["family"] = c["strategy"].map(FAMILY)
    return c


def daily_counts(sig: pd.DataFrame, by: str, date_col: str = "signal_date") -> pd.DataFrame:
    """Signals per trading day per `by` group (strategy / family), reindexed to BDAYS."""
    ct = sig.groupby([date_col, by]).size().unstack(by).reindex(BDAYS).fillna(0.0)
    return ct


def trailing(ct: pd.DataFrame, k: int) -> pd.DataFrame:
    return ct.rolling(k, min_periods=1).sum()


def attach_flow(trades: pd.DataFrame, sig: pd.DataFrame, date_col: str = "signal_date",
                strat_col: str = "strategy") -> pd.DataFrame:
    """Attach flow state variables to each trade, as of its signal date (inclusive)."""
    t = trades.copy()
    sig = sig.copy()
    sig["family"] = sig[strat_col].map(FAMILY)
    cs = daily_counts(sig, strat_col, date_col)
    cf = daily_counts(sig, "family", date_col)
    cb = cs.sum(axis=1).to_frame("book")
    strat_firing = sig.groupby(date_col)[strat_col].nunique().reindex(BDAYS).fillna(0.0)
    d = t["Signal Date"]
    for k, lab in [(1, "1"), (5, "5"), (21, "21"), (63, "63")]:
        ts, tf, tb = trailing(cs, k), trailing(cf, k), trailing(cb, k)
        t[f"s{lab}"] = [ts.at[dd, s] if s in ts.columns else 0.0 for dd, s in zip(d, t["Strategy"])]
        t[f"f{lab}"] = [tf.at[dd, f] if f in tf.columns else 0.0 for dd, f in zip(d, t["family"])]
        t[f"b{lab}"] = tb["book"].reindex(d).values
    # relative flow: trailing-21 count vs trailing-252 mean of 21d counts (era-normalised), strategy + family
    s21, f21 = trailing(cs, 21), trailing(cf, 21)
    s21n = s21 / s21.rolling(252, min_periods=63).mean().shift(21).replace(0, np.nan)
    f21n = f21 / f21.rolling(252, min_periods=63).mean().shift(21).replace(0, np.nan)
    t["s21_rel"] = [s21n.at[dd, s] if s in s21n.columns else np.nan for dd, s in zip(d, t["Strategy"])]
    t["f21_rel"] = [f21n.at[dd, f] if f in f21n.columns else np.nan for dd, f in zip(d, t["family"])]
    t["nstrat1"] = strat_firing.reindex(d).values
    t["nstrat5"] = strat_firing.rolling(5, min_periods=1).sum().reindex(d).values
    return t


def attach_open_legs(trades: pd.DataFrame) -> pd.DataFrame:
    """Open legs (fills) per strategy / family / book at the signal date, EXCLUDING legs signalled that day."""
    t = trades.copy()
    ent, ex = t["Entry Date"].values, t["ExitDate"].values
    d = t["Signal Date"].values
    strat, fam = t["Strategy"].values, t["family"].values
    n = len(t)
    open_s, open_f, open_b = np.zeros(n), np.zeros(n), np.zeros(n)
    # vectorised per strategy: for each trade, count others with entry <= d < exit (entered strictly before signal day)
    order = np.argsort(d)
    for i in range(n):
        m = (ent <= d[i]) & (ex >= d[i]) & (d < d[i])
        open_b[i] = m.sum()
        open_s[i] = (m & (strat == strat[i])).sum()
        open_f[i] = (m & (fam == fam[i])).sum()
    t["open_s"], t["open_f"], t["open_b"] = open_s, open_f, open_b
    return t


def episodes(dates: pd.Series, gap_td: int = 5) -> np.ndarray:
    """Episode id for a sorted-or-not series of dates: new episode when the gap to the previous date > gap_td bdays."""
    ds = pd.to_datetime(dates)
    order = np.argsort(ds.values)
    pos = pd.Series(np.searchsorted(BDAYS.values, ds.values[order]))
    new = (pos.diff().fillna(gap_td + 1) > gap_td).astype(int).cumsum().values
    out = np.empty(len(ds), dtype=int)
    out[order] = new
    return out


def cluster_boot_diff(x_hi: np.ndarray, c_hi: np.ndarray, x_lo: np.ndarray, c_lo: np.ndarray, n=1000, seed=0):
    """Cluster bootstrap (resample episodes) of mean(hi) - mean(lo). Returns diff, se, t, p_two_sided_boot."""
    rng = np.random.default_rng(seed)
    def boot(x, c):
        ids = np.unique(c)
        groups = [x[c == g] for g in ids]
        out = np.empty(n)
        for i in range(n):
            pick = rng.integers(0, len(groups), len(groups))
            v = np.concatenate([groups[j] for j in pick])
            out[i] = v.mean()
        return out
    if len(x_hi) < 5 or len(x_lo) < 5:
        return dict(diff=np.nan, se=np.nan, t=np.nan, p=np.nan)
    bh, bl = boot(x_hi, c_hi), boot(x_lo, c_lo)
    d = x_hi.mean() - x_lo.mean()
    se = float(np.std(bh - bl))
    return dict(diff=float(d), se=se, t=float(d / se) if se > 0 else np.nan, p=float(2 * min((bh - bl <= 0).mean(), (bh - bl >= 0).mean())))


def load_strategy_daily() -> tuple[pd.DataFrame, pd.Series]:
    import json
    sd = json.load(open(ROOT / "dist/data/strategy_daily.json"))
    dates = pd.to_datetime(sd["dates"])
    S = pd.DataFrame(sd["series"], index=dates).fillna(0.0)
    strat = S.T.groupby(S.columns.str.split("||", regex=False).str[0]).sum().T
    tot = pd.Series(sd["total_flat"], index=dates, dtype=float)
    return strat, tot


def family_daily(strat: pd.DataFrame) -> pd.DataFrame:
    fam = pd.DataFrame({f: strat[[s for s in strat.columns if FAMILY.get(s) == f]].sum(axis=1) for f in FAMILIES})
    fam["book"] = strat.sum(axis=1)
    return fam


def spearman(a, b):
    m = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(m) < 10:
        return np.nan
    return float(m["a"].rank().corr(m["b"].rank()))


def dd_stats(daily_pnl: pd.Series) -> dict:
    eq = daily_pnl.cumsum()
    dd = eq - eq.cummax()
    sd = daily_pnl.std()
    return dict(total=float(daily_pnl.sum()), sd_daily=float(sd), sharpe=float(daily_pnl.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan,
                maxdd=float(dd.min()), worst21=float(daily_pnl.rolling(21).sum().min()))


def realized_at_exit_daily(trades: pd.DataFrame, mult: np.ndarray | None = None) -> pd.Series:
    m = np.ones(len(trades)) if mult is None else mult
    p = pd.Series(trades["PnL"].values * m, index=trades["ExitDate"].values).groupby(level=0).sum()
    return p.reindex(pd.bdate_range(p.index.min(), p.index.max())).fillna(0.0)


def build_trade_mtm(trades: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-trade daily MTM vectors on the flat basis from master_prices closes; each vector reconciled to booked PnL.
    Returns (dates index, matrix [n_trades x n_days]) as a sparse-ish dict of arrays to save memory."""
    ticks = sorted(set(trades["Ticker"]))
    tbl = pq.read_table(ROOT / "data/master_prices.parquet", columns=["ticker", "date", "Close"],
                        filters=[("ticker", "in", ticks), ("date", ">=", pd.Timestamp("2002-12-01"))]).to_pandas()
    px = tbl.pivot(index="date", columns="ticker", values="Close").sort_index()
    px.index = pd.to_datetime(px.index)
    days = pd.bdate_range(trades["Entry Date"].min(), trades["ExitDate"].max())
    px = px.reindex(days).ffill()
    day_pos = {d: i for i, d in enumerate(days)}
    out = np.zeros((len(trades), len(days)), dtype=np.float32)
    sign = np.where(trades["Direction"].values == "Long", 1.0, -1.0)
    for i, (tk, e, x, sh, ep, pnl) in enumerate(zip(trades["Ticker"], trades["Entry Date"], trades["ExitDate"],
                                                     trades["Shares"], trades["EntryPrice"], trades["PnL"])):
        a, b = day_pos.get(e), day_pos.get(x)
        if a is None or b is None or tk not in px.columns:
            if b is not None:
                out[i, b] = pnl
            continue
        c = px[tk].values[a:b + 1]
        if np.isnan(c).any() or sh == 0:
            out[i, b] = pnl
            continue
        v = np.empty(b - a + 1)
        v[0] = (c[0] - ep) * sh * sign[i]
        v[1:] = np.diff(c) * sh * sign[i]
        v[-1] += pnl - v.sum()   # reconcile to booked exit
        out[i, a:b + 1] = v
    return days, out
