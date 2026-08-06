"""Defined-risk short vol: LONG SVXY over the event/seasonal windows.

SVXY is the -0.5x VIX short-term futures ETP (post 2018-02-27; -1x before,
which died in Volmageddon — only the -0.5x regime is used live). Long SVXY
= short vol with loss BOUNDED at the position, unlike a naked UVXY short.

Sections:
  1. Instrument check: real SVXY (yfinance) vs synthetic (-1/3 x UVXY ret)
     on the overlap, then a spliced series (-0.5x regime only, 2018-03+,
     plus synthetic back-extension 2011-10..2018-02 for context).
  2. Windows on the -0.5x basis: pre-FOMC (all/ex-mid/mid), Nov1-Dec31,
     Dec opex -> year-end.
  3. THE DIVERSIFICATION QUESTION: within pre-FOMC windows, quadrant the
     joint (SPY window ret, short-vol window ret). How often does short
     vol WIN while SPY is DOWN? Window-PnL correlation, and combined
     T1-equity + vol-leg stats.

Run: python scratch/svxy_defined_risk_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from macro_calendar import event_dates  # noqa: E402

CACHE = ROOT / "scratch" / "svxy_yf.parquet"


def load_master(tkr: str) -> pd.DataFrame:
    mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                         columns=["ticker", "date", "Close"])
    df = mp[mp["ticker"] == tkr].set_index("date").sort_index()[["Close"]]
    df.index = pd.to_datetime(df.index).normalize()
    return df[~df.index.duplicated(keep="last")]


def load_svxy() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    import yfinance as yf
    raw = yf.download("SVXY", start="2011-10-01", auto_adjust=True,
                      progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.xs("SVXY", level="Ticker", axis=1)
    raw.columns = [c.capitalize() for c in raw.columns]
    df = raw[["Close"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.to_parquet(CACHE)
    return df


def stats(x: pd.Series, label: str) -> str:
    x = x.dropna()
    if len(x) < 3:
        return f"{label:46s} N={len(x)}"
    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    return (f"{label:46s} {x.mean()*1e4:+8.1f} bps  t {t:+5.2f}  N {len(x):3d}"
            f"  hit {(x>0).mean():.2f}  worst {x.min()*1e4:+7.0f}")


def win_ret(r: pd.Series, anchor: pd.Timestamp, a: int, b: int) -> float:
    """Compound r over td offsets a..b around anchor's session."""
    idx = r.index
    p = idx.searchsorted(anchor)
    lo, hi = p + a, p + b
    if lo < 0 or hi >= len(idx) or p >= len(idx):
        return np.nan
    seg = r.iloc[lo:hi + 1]
    return float((1 + seg).prod() - 1)


uvxy = load_master("UVXY")
spy = load_master("SPY")
svxy_real = load_svxy()

u_ret = uvxy["Close"].pct_change()
syn = -u_ret / 3.0                     # synthetic -0.5x from the +1.5x ETP
real = svxy_real["Close"].pct_change()

print("=" * 100)
print("1. INSTRUMENT CHECK")
print("=" * 100)
ov = pd.concat([syn.rename("syn"), real.rename("real")], axis=1).dropna()
ov_post = ov.loc["2018-03-01":]
print(f"synthetic vs real SVXY daily corr, -0.5x regime (2018-03+): "
      f"{ov_post['syn'].corr(ov_post['real']):.4f}  "
      f"(N={len(ov_post)}, mean diff "
      f"{(ov_post['real']-ov_post['syn']).mean()*1e4:+.1f} bps/day)")
# splice: real where -0.5x regime exists, synthetic before
sv = pd.concat([syn.loc[:"2018-02-27"], real.loc["2018-02-28":]]).sort_index()
sv = sv[~sv.index.duplicated(keep="last")]
print(f"spliced short-vol series: {sv.index.min().date()} .. "
      f"{sv.index.max().date()}  all-days {sv.mean()*1e4:+.1f} bps/day "
      f"(the harvest side of the decay)")

print()
print("=" * 100)
print("2. WINDOWS — LONG SVXY (-0.5x basis)")
print("=" * 100)
fomc = event_dates("fomc_decision")
fomc = fomc[(fomc >= sv.index.min()) & (fomc <= sv.index.max())]
w = pd.Series([win_ret(sv, d, -3, 0) for d in fomc], index=fomc)
print(stats(w, "pre-FOMC (-3..0) ALL"))
print(stats(w[[d.year % 4 != 2 for d in w.index]], "pre-FOMC ex-midterm"))
print(stats(w[[d.year % 4 == 2 for d in w.index]], "pre-FOMC midterm"))
w18 = w.loc["2018-03-01":]
print(stats(w18, "pre-FOMC ALL, real -0.5x era only (2018-03+)"))

rows = []
for y in range(2012, 2026):
    seg = sv.loc[f"{y}-11-01":f"{y}-12-31"]
    if len(seg) > 10:
        rows.append((y, float((1 + seg).prod() - 1)))
print(stats(pd.Series([v for _, v in rows]), "Nov 1 -> Dec 31 seasonal"))
print("   " + " ".join(f"{y}:{v*100:+.0f}%" for y, v in rows))

opex = event_dates("opex")
dec_opex = [d for d in opex if d.month == 12
            and sv.index.min() <= d <= sv.index.max()]
wd = []
for d in dec_opex:
    p = sv.index.searchsorted(d)
    ye = sv.index.searchsorted(pd.Timestamp(d.year, 12, 31), side="right") - 1
    if ye > p:
        seg = sv.iloc[p + 1: ye + 1]
        wd.append(float((1 + seg).prod() - 1))
print(stats(pd.Series(wd), "Dec opex -> year-end"))

print()
print("=" * 100)
print("3. DIVERSIFICATION vs THE EQUITY SLEEVE (pre-FOMC windows)")
print("=" * 100)
s_ret = spy["Close"].pct_change()
recs = []
for d in fomc:
    recs.append({"date": d, "midterm": d.year % 4 == 2,
                 "spy": win_ret(s_ret, d, -3, 0),
                 "vol": win_ret(sv, d, -3, 0)})
j = pd.DataFrame(recs).dropna()
print(f"window PnL correlation (SPY vs short-vol): "
      f"{j['spy'].corr(j['vol']):+.3f}   rank "
      f"{j['spy'].corr(j['vol'], method='spearman'):+.3f}")
dn = j[j.spy < 0]
print(f"SPY DOWN windows: {len(dn)} of {len(j)}; short vol still POSITIVE "
      f"in {int((dn.vol > 0).sum())} of them "
      f"({(dn.vol > 0).mean():.0%}) — McKinley's point, quantified")
print(stats(dn["vol"], "  short-vol ret | SPY down window"))
up = j[j.spy >= 0]
print(stats(up["vol"], "  short-vol ret | SPY up window"))

ex = j[~j.midterm]
eq_leg = ex["spy"]
vol_leg = ex["vol"]
for wv in (0.0, 0.25, 0.5):
    combo = (1 - wv) * eq_leg + wv * vol_leg
    t = combo.mean() / (combo.std(ddof=1) / np.sqrt(len(combo)))
    print(f"  ex-midterm combo {1-wv:.0%} SPY / {wv:.0%} SVXY: "
          f"{combo.mean()*1e4:+7.1f} bps  t {t:+5.2f}  "
          f"worst {combo.min()*1e4:+7.0f}  hit {(combo>0).mean():.2f}")
