"""C7 -- parabolic extension above the 200d, CROSS-SECTIONAL (short side).

Built as the peer group from the start, per the 2026-08-13 reference-class
lesson: never "fade MU", always "what does the >=Xth-pctile-own-extension
state pay across every name that has ever been in it".

State (point-in-time, no lookahead):
    ext        = Close / SMA200 - 1
    ext_pctile = EXPANDING percentile of that name's own ext history
                 (min 756 obs = 3y) -- the recon's "97.7th pctile of its own
                 history" made PIT.
Trigger: ext_pctile >= 95 (variants 90 / 97.5 / 99), ext > 0.
Forward: lag=1, h=1..10, SHORT so returns are negated at report time.
Decluster per NAME at 21 td (an extension state persists for weeks).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import strategy_config as sc  # noqa: E402
from pitch_lab import PRICES_PATH, declusters, show, summarize, sign_test  # noqa: E402

BAD = {"SOXS"}                       # corrupt upstream pre-2026-05-26
MIN_HIST = 756
GAP = 21


def panel(tickers: list[str]) -> pd.DataFrame:
    mp = pd.read_parquet(PRICES_PATH, columns=["date", "ticker", "Close"])
    mp = mp[mp["ticker"].isin(tickers)]
    mp["date"] = pd.to_datetime(mp["date"])
    p = mp.pivot_table(index="date", columns="ticker", values="Close",
                       aggfunc="last").sort_index()
    return p


uni = sorted(set(sc.LIQUID_PLUS_COMMODITIES) - BAD)
px = panel(uni)
px = px.loc[:, px.notna().sum() >= MIN_HIST + 300]
print(f"universe {px.shape[1]} names, {px.index[0].date()} .. "
      f"{px.index[-1].date()}, {len(px)} sessions")

sma200 = px.rolling(200).mean()
ext = px / sma200 - 1.0
extp = ext.expanding(MIN_HIST).rank(pct=True) * 100.0

# forward returns lag=1, per name
FWD = {h: px.shift(-(1 + h)) / px.shift(-1) - 1.0 for h in range(1, 11)}


def collect(thresh: float) -> dict[int, pd.DataFrame]:
    """Per (name, date) trigger rows, declustered per name, all h."""
    m = (extp >= thresh) & (ext > 0)
    out = {}
    for h in range(1, 11):
        recs = []
        f = FWD[h]
        for tkr in px.columns:
            mt = m[tkr].fillna(False) & f[tkr].notna()
            d = px.index[mt.values]
            if len(d) == 0:
                continue
            epi = declusters(d, GAP, px.index)
            for dt in epi:
                recs.append((tkr, dt, f.at[dt, tkr]))
        out[h] = pd.DataFrame(recs, columns=["tkr", "date", "ret"])
    return out


def report(df: pd.DataFrame, h: int, thresh: float, base: pd.Series) -> dict:
    """SHORT the extended name: pnl = -ret."""
    pnl = -df["ret"].values
    r = summarize(pnl, f"thresh {thresh:.1f} h={h} SHORT")
    r["n_names"] = df["tkr"].nunique()
    r["ctrl_all_short_pct"] = round(-100 * base.mean(), 3)
    r["edge_pct"] = round(r["mean_pct"] - r["ctrl_all_short_pct"], 3)
    return r


base_by_h = {h: FWD[h].stack().dropna() for h in range(1, 11)}

print("\n### 1. horizon scan, cross-sectional, SHORT the extended name "
      "(pctile >= 95) ###")
c95 = collect(95.0)
rows = [report(c95[h], h, 95.0, base_by_h[h]) for h in range(1, 11)]
show(rows)

print("\n### 2. threshold neighbours at h=3 and h=5 ###")
for h in (3, 5, 10):
    rows = []
    for t in (90.0, 95.0, 97.5, 99.0):
        c = collect(t)
        rows.append(report(c[h], h, t, base_by_h[h]))
    show(rows, f"h={h}")

print("\n### 3. cross-name distribution at pctile>=95 (the reference class) ###")
for h in (3, 5, 10):
    d = c95[h].copy()
    d["pnl"] = -d["ret"]
    g = d.groupby("tkr")["pnl"].agg(["count", "mean"])
    g = g[g["count"] >= 5]
    print(f"h={h}: {len(g)} names with >=5 episodes | "
          f"share of names with positive SHORT mean = "
          f"{100*(g['mean'] > 0).mean():.1f}% | "
          f"median name mean = {100*g['mean'].median():+.3f}% | "
          f"pooled = {100*d['pnl'].mean():+.3f}%")
    top = g.sort_values("mean", ascending=False)
    print("   best 5 names:", {k: round(100*v, 2) for k, v in
                               top["mean"].head(5).items()})
    print("   worst 5 names:", {k: round(100*v, 2) for k, v in
                                top["mean"].tail(5).items()})

print("\n### 4. era split, pctile>=95 ###")
for h in (3, 5, 10):
    d = c95[h]
    pre = -d.loc[d["date"] < "2018-01-01", "ret"].values
    post = -d.loc[d["date"] >= "2018-01-01", "ret"].values
    show([summarize(pre, f"h={h} pre-2018"), summarize(post, f"h={h} 2018+")])

print("\n### 5. by year, h=5 ###")
d = c95[5].copy()
d["pnl"] = -d["ret"]
yr = d.groupby(d["date"].dt.year)["pnl"].agg(["count", "mean"])
print((yr.assign(mean_pct=(100*yr["mean"]).round(3))[["count", "mean_pct"]]
       ).to_string())
pos_yrs = int((yr["mean"] > 0).sum())
print(f"positive years {pos_yrs}/{len(yr)}, sign p="
      f"{sign_test(pos_yrs, len(yr)):.4f}")

print("\n### 6. ETF vs single-stock split (registry adjacency a) h=5 ###")
etf_like = {t for t in px.columns if len(t) <= 4 and t.isupper()}
# use a real classifier: names in the SECTOR/commodity ETF set
try:
    etfs = set(sc.SECTOR_ETFS)
except Exception:
    etfs = set()
known_etf = {"SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "EWJ", "EWZ", "FXI",
             "GLD", "SLV", "USO", "UNG", "TLT", "IEF", "LQD", "HYG", "GDX",
             "XLE", "XLF", "XLK", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB",
             "XLRE", "XLC", "SMH", "XRT", "XOP", "XBI", "IBB", "IHI", "KRE",
             "XHB", "ITB", "JETS", "TAN", "ICLN", "ARKK", "DBC", "UUP", "SVXY",
             "VNQ", "VOO", "SPLG", "SLX", "COPX", "SIL", "PPLT", "PALL"}
etfs |= known_etf
d = c95[5].copy()
d["pnl"] = -d["ret"]
d["is_etf"] = d["tkr"].isin(etfs)
show([summarize(d.loc[d["is_etf"], "pnl"].values, "ETFs"),
      summarize(d.loc[~d["is_etf"], "pnl"].values, "single stocks")])

print("\n### 7. MU-scale extension only: ext >= 0.50 (the live magnitude) ###")
for h in (3, 5, 10):
    m = (extp >= 95.0) & (ext >= 0.50)
    recs = []
    f = FWD[h]
    for tkr in px.columns:
        mt = m[tkr].fillna(False) & f[tkr].notna()
        dts = px.index[mt.values]
        if len(dts) == 0:
            continue
        for dt in declusters(dts, GAP, px.index):
            recs.append((tkr, dt, f.at[dt, tkr]))
    dd = pd.DataFrame(recs, columns=["tkr", "date", "ret"])
    if dd.empty:
        print(f"h={h}: no rows")
        continue
    pnl = -dd["ret"].values
    r = summarize(pnl, f"ext>=50% & pctile>=95, h={h}, SHORT")
    r["n_names"] = dd["tkr"].nunique()
    show([r])
    print("   years:", dict(dd.groupby(dd['date'].dt.year).size()))

print("\n### 8. cost: single-name SHORT ~10 bps round trip + borrow. "
      "Need >= 5x = ~+0.50% per episode. ###")
