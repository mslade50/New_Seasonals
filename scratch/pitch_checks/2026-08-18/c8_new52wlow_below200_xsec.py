"""C8 -- new 52w low AND >= 20% below the 200d, CROSS-SECTIONAL long.

Same design rule as C7: build the peer group, not "buy NKE".

State (point-in-time):
    at_low  = Close == trailing-252 min (a NEW 52w low printed today)
    deep    = Close / SMA200 - 1 <= -0.20
Trigger = at_low & deep. Forward lag=1, h=1..10, LONG. Decluster per name
at 21 td.

Also answers, with numbers not prose:
  - is this state inside or outside the book's dip-buy gates?
  - the 2026-08-14 ALPHABETICAL PLACEBO for a 4-name tradeable cut
  - survivorship: the worst possible cell for it. Direction + a bound.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import strategy_config as sc  # noqa: E402
from pitch_lab import PRICES_PATH, declusters, show, summarize, sign_test  # noqa: E402

BAD = {"SOXS"}
GAP = 21
DEEP = -0.20


def panel(tickers: list[str]) -> pd.DataFrame:
    mp = pd.read_parquet(PRICES_PATH, columns=["date", "ticker", "Close"])
    mp = mp[mp["ticker"].isin(tickers)]
    mp["date"] = pd.to_datetime(mp["date"])
    return mp.pivot_table(index="date", columns="ticker", values="Close",
                          aggfunc="last").sort_index()


uni = sorted(set(sc.LIQUID_PLUS_COMMODITIES) - BAD)
px = panel(uni)
px = px.loc[:, px.notna().sum() >= 1000]
print(f"universe {px.shape[1]} names, {px.index[0].date()} .. "
      f"{px.index[-1].date()}, {len(px)} sessions")

sma200 = px.rolling(200).mean()
low252 = px.rolling(252).min()
dist200 = px / sma200 - 1.0
at_low = px <= low252 * 1.0000001
deep = dist200 <= DEEP
r252 = px.pct_change(252)
rank252 = r252.rolling(252).rank(pct=True) * 100.0     # the book's perf filter

FWD = {h: px.shift(-(1 + h)) / px.shift(-1) - 1.0 for h in range(1, 11)}


def collect(deep_thresh: float, need_low: bool = True) -> dict[int, pd.DataFrame]:
    m = (at_low if need_low else pd.DataFrame(True, index=px.index,
                                              columns=px.columns))
    m = m & (dist200 <= deep_thresh)
    out = {}
    for h in range(1, 11):
        recs, f = [], FWD[h]
        for tkr in px.columns:
            mt = m[tkr].fillna(False) & f[tkr].notna()
            d = px.index[mt.values]
            if len(d) == 0:
                continue
            for dt in declusters(d, GAP, px.index):
                recs.append((tkr, dt, f.at[dt, tkr], dist200.at[dt, tkr],
                             rank252.at[dt, tkr]))
        out[h] = pd.DataFrame(recs, columns=["tkr", "date", "ret", "d200",
                                             "r252"])
    return out


base_by_h = {h: FWD[h].stack().dropna() for h in range(1, 11)}
c = collect(DEEP)

print("\n### 1. horizon scan, LONG the new 52w low >=20% below the 200d ###")
rows = []
for h in range(1, 11):
    d = c[h]
    r = summarize(d["ret"].values, f"h={h} LONG")
    r["n_names"] = d["tkr"].nunique()
    r["ctrl_all_pct"] = round(100 * base_by_h[h].mean(), 3)
    r["edge_pct"] = round(r["mean_pct"] - r["ctrl_all_pct"], 3)
    rows.append(r)
show(rows)

print("\n### 2. threshold neighbours on the depth gate ###")
for h in (3, 5, 10):
    rows = []
    for t in (-0.10, -0.15, -0.20, -0.25, -0.30):
        cc = collect(t)[h]
        if cc.empty:
            continue
        r = summarize(cc["ret"].values, f"d200<={t:+.2f} h={h}")
        r["n_names"] = cc["tkr"].nunique()
        rows.append(r)
    # and the 52w-low leg removed, to attribute the gate
    cc = collect(DEEP, need_low=False)[h]
    r = summarize(cc["ret"].values, f"deep ONLY (no 52w low) h={h}")
    r["n_names"] = cc["tkr"].nunique()
    rows.append(r)
    show(rows, f"h={h}")

print("\n### 3. cross-name distribution, the reference class ###")
for h in (3, 5, 10):
    d = c[h]
    g = d.groupby("tkr")["ret"].agg(["count", "mean"])
    g5 = g[g["count"] >= 5]
    print(f"h={h}: N={len(d)} episodes over {d['tkr'].nunique()} names | "
          f"{len(g5)} names with >=5 | share positive-mean names "
          f"{100*(g5['mean'] > 0).mean():.1f}% | median name mean "
          f"{100*g5['mean'].median():+.3f}% | pooled {100*d['ret'].mean():+.3f}%")

print("\n### 4. era split + by year ###")
for h in (3, 5, 10):
    d = c[h]
    show([summarize(d.loc[d["date"] < "2018-01-01", "ret"].values,
                    f"h={h} pre-2018"),
          summarize(d.loc[d["date"] >= "2018-01-01", "ret"].values,
                    f"h={h} 2018+")])
d = c[5]
yr = d.groupby(d["date"].dt.year)["ret"].agg(["count", "mean"])
print((yr.assign(mean_pct=(100*yr["mean"]).round(3))[["count", "mean_pct"]]
       ).to_string())
pos = int((yr["mean"] > 0).sum())
print(f"positive years {pos}/{len(yr)}, sign p={sign_test(pos, len(yr)):.4f}")
print("episodes by year concentration: top year share of total return = "
      f"{100*(yr['count']*yr['mean']).max()/ (yr['count']*yr['mean']).sum():.0f}%")

print("\n### 5. BOOK GATES: can any book dip-buy fire in this state? ###")
d = c[5]
print(f"  trigger rows' 252d perf rank: mean {d['r252'].mean():.1f}, "
      f"median {d['r252'].median():.1f}, "
      f"p95 {d['r252'].quantile(0.95):.1f}, "
      f"share >= 50 (the OLV / St OS Sznl / 52wh gate floor): "
      f"{100*(d['r252'] >= 50).mean():.2f}%")
print(f"  share >= 65 (LT Trend ST OS / Sector BO floor): "
      f"{100*(d['r252'] >= 65).mean():.2f}%")
print(f"  share above the 200d (Monthly Weak Close 'Price > 200 SMA', "
      f"Monday Dip 200d-consec-15, LT Trend 200d-consec-50): "
      f"{100*(d['d200'] > 0).mean():.2f}% (0 by construction, d200<=-20%)")
print(f"  median depth below the 200d at trigger: {100*d['d200'].median():.1f}%")

print("\n### 6. ALPHABETICAL PLACEBO (4-name tradeable cut), h=5 and h=10 ###")
for h in (3, 5, 10):
    m = at_low & deep
    f = FWD[h]
    sig_rows, alpha_rows, dates_used = [], [], 0
    last_used = -10**9
    pos = pd.Series(range(len(px.index)), index=px.index)
    for dt in px.index:
        q = [t for t in px.columns
             if bool(m.at[dt, t]) and pd.notna(f.at[dt, t])]
        if not q:
            continue
        if pos[dt] - last_used < GAP:      # decluster at the DATE level
            continue
        last_used = pos[dt]
        dates_used += 1
        # signal rule: the 4 DEEPEST below the 200d
        deepest = sorted(q, key=lambda t: dist200.at[dt, t])[:4]
        alpha = sorted(q)[:4]
        sig_rows.append(np.mean([f.at[dt, t] for t in deepest]))
        alpha_rows.append(np.mean([f.at[dt, t] for t in alpha]))
    show([summarize(np.array(sig_rows), f"h={h} SIGNAL 4 deepest (N={dates_used} dates)"),
          summarize(np.array(alpha_rows), f"h={h} PLACEBO 4 alphabetical")])

print("\n### 7. SURVIVORSHIP -- the direction and a bound ###")
print("  master_prices holds only tickers in TODAY's universe files. Every")
print("  name that printed a new 52w low 20%+ below its 200d and then went")
print("  to zero / was delisted is ABSENT. That omission is entirely on the")
print("  LOSING side of a long, so the pooled long mean below is an UPPER")
print("  BOUND. CLAUDE.md quantifies the same bias at 21 of 22 major 2020s")
print("  delistings absent from the ledger universe.")
print("\n### 8. cost: single-name long ~10 bps round trip; a 4-name basket")
print("  is 4 x 10 bps of the basket's own notional = 10 bps on the basket.")
print("  Need >= 5x = ~+0.50% per episode. ###")
