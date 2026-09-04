"""Trend sleeve universe + exhaustion-overlay study (McKinley's 3 questions).

Q1: do sector ETFs / more international equity ETFs help?
Q2: TLT+LQD in, USO out — cost/benefit?
Q3: scale-down when trailing 252d AND 21d return percentiles both > 95?

Engine copied verbatim from scratch/ultracode_research/tf_backtest.py (verified
spec): combo = 12-1 mom AND 10m MA, long/flat, inverse-vol slots capped 20%,
5 bps/side, weights at month-end t held over t+1. Universe-variant selection on
the same history adds one in-sample degree of freedom — read deltas, not levels.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COST_PER_SIDE = 0.0005
CAP = 0.20

CORE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "FXI", "VNQ", "GLD", "SLV", "DBC", "UUP"]
SECTORS = ["XLE", "XLF", "XLK", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE", "XLC"]
INTL = ["EWJ", "EWY", "EWT", "EWZ", "INDA"]
VARIANTS = {
    "A pilot ex-bonds 12 (live)": CORE + ["USO"],
    "B core-USO +TLT+LQD (13)": CORE + ["TLT", "LQD"],
    "C = B + 11 sectors (24)": CORE + ["TLT", "LQD"] + SECTORS,
    "D = B + 5 intl (18)": CORE + ["TLT", "LQD"] + INTL,
    "E = B + sectors + intl (29)": CORE + ["TLT", "LQD"] + SECTORS + INTL,
    "F full16 (reference)": CORE + ["USO", "TLT", "IEF", "LQD", "HYG"],
}
ALL = sorted({t for v in VARIANTS.values() for t in v})


def load_prices():
    mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                         columns=["ticker", "date", "Close"])
    sub = mp[mp.ticker.isin(ALL + ["^IRX"])]
    wide = sub.pivot(index="date", columns="ticker", values="Close").sort_index()
    wide.index = pd.to_datetime(wide.index)
    irx = wide.pop("^IRX").ffill()
    return wide.astype(float), irx


def build_combo(px):
    m = px.resample("ME").last()
    mom12_1 = m.shift(1) / m.shift(12) - 1.0
    ma10 = m - m.rolling(10).mean()
    elig = m.notna() & m.shift(12).notna()
    return ((mom12_1 > 0) & (ma10 > 0)).where(elig)


def exhaustion_mask(px):
    """True when BOTH trailing 252d and 21d returns are above their own
    expanding 95th percentile (>= 36 months of history) at month-end."""
    m252 = (px / px.shift(252) - 1.0).resample("ME").last()
    m21 = (px / px.shift(21) - 1.0).resample("ME").last()

    def expanding_pct(df):
        def _p(s):
            out = pd.Series(np.nan, index=s.index)
            v = s.values
            for i in range(len(v)):
                if np.isnan(v[i]):
                    continue
                hist = v[:i + 1]
                hist = hist[~np.isnan(hist)]
                if len(hist) >= 36:
                    out.iloc[i] = (hist <= v[i]).mean()
            return out
        return df.apply(_p)

    return (expanding_pct(m252) > 0.95) & (expanding_pct(m21) > 0.95)


def run(px, irx, sig, universe, exh=None, exh_mult=0.5):
    px = px[[t for t in universe if t in px.columns]]
    sig = sig[px.columns]
    m = px.resample("ME").last()
    aret = m.pct_change()
    rf = (irx.resample("ME").last() / 100.0).reindex(aret.index).ffill() / 12.0
    dret = px.pct_change()
    vol_m = (dret.rolling(63).std() * np.sqrt(252)).resample("ME").last().clip(lower=0.04)
    inv = (1.0 / vol_m).where(sig.notna()).fillna(0.0)
    w = inv.div(inv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0).clip(upper=CAP)
    w = w * sig.fillna(False).astype(float)
    if exh is not None:
        w = w * exh[px.columns].reindex(w.index).fillna(False).map(
            lambda x: exh_mult if x else 1.0) if False else \
            w * np.where(exh[px.columns].reindex(w.index).fillna(False), exh_mult, 1.0)
    w_held = w.shift(1).fillna(0.0)
    port = (w_held * aret).sum(axis=1)
    cash = (1.0 - w_held.sum(axis=1)) * rf.shift(1).fillna(0.0)
    cost = (w - w.shift(1)).abs().sum(axis=1).shift(1).fillna(0.0) * COST_PER_SIDE
    net = port + cash - cost
    n_elig = sig.notna().sum(axis=1)
    start = n_elig[n_elig >= 3].index.min()
    held = (w_held > 0).sum(axis=1)
    return (pd.DataFrame({"net": net, "rf": rf, "held": held,
                          "exp": w_held.sum(axis=1)})
            .loc[lambda d: d.index >= start])


def stats(d, since=None):
    r = d["net"].dropna()
    if since:
        r = r[r.index >= since]
    curve = (1 + r).cumprod()
    yrs = len(r) / 12.0
    ex = r - d["rf"].reindex(r.index).fillna(0.0)
    return {
        "N": len(r),
        "CAGR": (curve.iloc[-1] ** (1 / yrs) - 1) * 100,
        "Vol": r.std() * np.sqrt(12) * 100,
        "Sharpe": ex.mean() / r.std() * np.sqrt(12),
        "MaxDD": (curve / curve.cummax() - 1).min() * 100,
        "held": d["held"].reindex(r.index).mean(),
        "exp%": d["exp"].reindex(r.index).mean() * 100,
    }


def yr_ret(d, yr):
    r = d["net"][d["net"].index.year == yr]
    return ((1 + r).prod() - 1) * 100 if len(r) else np.nan


px, irx = load_prices()
sig = build_combo(px)
exh = exhaustion_mask(px)

# book + fragility for fit columns
led = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
led["Exit Date"] = pd.to_datetime(led["Exit Date"])
book = (led.set_index("Exit Date")["PnL_flat_750k"]
        .groupby(pd.Grouper(freq="ME")).sum() / 750000)
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
fma = frag["63d"].dropna().rolling(10, min_periods=1).mean()
hifrag = fma.groupby(pd.Grouper(freq="ME")).mean() >= 50

print(f"exhaustion months (any asset flagged): "
      f"{exh.any(axis=1).mean()*100:.0f}% of months; "
      f"avg flagged when any: {exh.sum(axis=1)[exh.any(axis=1)].mean():.1f}")
print()
hdr = (f"{'variant':<30} {'N':>4} {'CAGR':>6} {'Vol':>6} {'Sharpe':>6} {'MaxDD':>7} "
       f"{'held':>5} {'exp%':>5} {'2008':>6} {'2022':>6} {'corrBk':>6} {'hiFrag':>7} {'S16+':>5}")
print(hdr)
for name, uni in VARIANTS.items():
    d = run(px, irx, sig, uni)
    s = stats(d)
    common = d["net"].dropna().index.intersection(book.index)
    corr = d["net"].reindex(common).corr(book.reindex(common))
    hf_idx = [i for i in common if i in hifrag.index and hifrag.loc[i] and i.year >= 2016]
    hf = d["net"].reindex(hf_idx).mean() * 100 if hf_idx else np.nan
    s16 = stats(d, since="2016-01-01")["Sharpe"]
    print(f"{name:<30} {s['N']:>4} {s['CAGR']:>5.2f}% {s['Vol']:>5.2f}% {s['Sharpe']:>6.2f} "
          f"{s['MaxDD']:>6.1f}% {s['held']:>5.1f} {s['exp%']:>5.0f} "
          f"{yr_ret(d, 2008):>5.1f}% {yr_ret(d, 2022):>5.1f}% {corr:>6.2f} {hf:>6.2f}% {s16:>5.2f}")

print("\n--- exhaustion overlay (both 252d & 21d expanding pctile > 95 -> scale ON weight) ---")
for name in ["B core-USO +TLT+LQD (13)", "C = B + 11 sectors (24)", "E = B + sectors + intl (29)"]:
    uni = VARIANTS[name]
    for mult, lbl in [(1.0, "no overlay"), (0.5, "0.5x flagged"), (0.0, "exit flagged")]:
        d = run(px, irx, sig, uni, exh=exh if mult < 1.0 else None, exh_mult=mult)
        s = stats(d)
        s16 = stats(d, since="2016-01-01")["Sharpe"]
        print(f"{name:<30} {lbl:<13} CAGR {s['CAGR']:5.2f}%  Vol {s['Vol']:5.2f}%  "
              f"Sharpe {s['Sharpe']:5.2f}  MaxDD {s['MaxDD']:6.1f}%  S16+ {s16:5.2f}")
