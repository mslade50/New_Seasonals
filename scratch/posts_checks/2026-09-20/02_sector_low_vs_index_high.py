"""Posts check (2026-09-20), part 2: a sector ETF at a 252-session closing LOW
while SPY sits within 3% of its own 252-session closing high.

Script 01 found the XLU-only version of this has happened exactly once in
26 years, and that once is Friday 2026-09-18. n=1 is a fact, not an edge, so
this widens the same state across all 11 SPDR sector ETFs to get a record.

E. Inventory. Every XLU close at a 252-session low since 2000, with SPY's
   distance from its own 252-session closing high on that session. This is
   what makes "the first time" checkable.
F. POOLED cell. Any of the 11 SPDRs closing at a 252-session low while SPY is
   within 3% of its 252-session high. Novelty-declustered 21 sessions PER
   TICKER. Forward sector return and the sector-minus-SPY spread at
   h=5/10/21, lag=1 (enter MOC the session after the signal), against three
   controls: all sector-sessions, sector sessions at a 252d low with SPY NOT
   near a high (the discriminating control), and SPY itself.
G. Era split and a per-ticker breakdown of F, plus the episode table.
H. XLU's own 21-session return rank at or below 3 while SPY is within 3% of
   its high, as the softer XLU-only version.

XLE/XLF/XLI/XLK/XLP/XLU/XLV/XLY/XLB all start 1998-12; XLRE 2015-10 and
XLC 2018-06, so the pool is unbalanced by construction and the per-ticker
table is printed for that reason.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import (  # noqa: E402
    fwd_lag, load_prices, pct_rank, rolling_on_valid, sign_test, summarize,
    wilder_atr,
)

ASOF = pd.Timestamp("2026-09-18")
ERA = "2018-01-01"
SECTORS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU",
           "XLV", "XLY"]
TK = SECTORS + ["SPY"]

raw_px = load_prices(TK)
nyse = raw_px["SPY"]["Close"].dropna().index
nyse = pd.DatetimeIndex(nyse[nyse <= ASOF])
C = {t: raw_px[t]["Close"].astype(float).reindex(nyse) for t in TK}
POS = pd.Series(range(len(nyse)), index=nyse)

spy_hi = rolling_on_valid(C["SPY"], lambda x: x.rolling(252).max())
spy_dist = C["SPY"] / spy_hi - 1.0
near_hi = (spy_dist >= -0.03)

print(f"calendar {nyse[0].date()} .. {nyse[-1].date()}  n={len(nyse)}")
for t in TK:
    v = C[t].dropna()
    print(f"  {t}: {v.index[0].date()} .. {v.index[-1].date()}  n={len(v)}  "
          f"last {v.iloc[-1]:.2f}")
print(f"SPY is {100*spy_dist.iloc[-1]:+.2f}% from its 252d closing high today")


def rec(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return int((v > 0).sum()), int((v < 0).sum()), len(v)


def novelty_t(trig, win=21):
    tset = set(pd.DatetimeIndex(trig))
    keep = []
    for d in sorted(tset):
        p = int(POS[d])
        if not any(x in tset for x in nyse[max(0, p - win):p]):
            keep.append(d)
    return pd.DatetimeIndex(keep)


def block(label, vals):
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        print(f"  {label}: n=0")
        return
    up, dn, n = rec(vals)
    sm = summarize(vals)
    # sign_test uses exact rational arithmetic, so it is only run on the
    # conditional cells; a control of tens of thousands of days would hang it.
    sp = (f"p_up {sign_test(up, n):.4f} p_dn {sign_test(dn, n):.4f} "
          if n <= 2000 else "p_up n/a  p_dn n/a  ")
    print(f"  {label}: n={n} {up}-{dn} mean {sm['mean_pct']:+.3f}% "
          f"med {sm['median_pct']:+.3f}% hit {sm['hit']:.1f}% t {sm['t']:+.2f} "
          f"{sp}worst {sm['worst_pct']:+.2f}% best {sm['best_pct']:+.2f}%")


# =========================================================================
# E. every XLU close at a 252-session low, and where SPY was
# =========================================================================
print("\n" + "=" * 78)
print("=== E. every XLU close at a 252-session closing low since 2000 ===")
xlu_lo = rolling_on_valid(C["XLU"], lambda x: x.rolling(252).min())
at_low = (C["XLU"] <= xlu_lo) & C["XLU"].notna() & xlu_lo.notna()
lows = pd.DatetimeIndex(nyse[at_low.fillna(False)])
print(f"   XLU closed at a 252d low on {len(lows)} sessions "
      f"({lows[0].date()} .. {lows[-1].date()})")
ep = novelty_t(lows, 21)
print(f"   distinct episodes (21-session novelty): {len(ep)}")
d_on_lows = spy_dist.reindex(lows).dropna()
print(f"   SPY distance from its own 252d high on those {len(d_on_lows)} sessions:")
print(f"     best (closest to a high): {100*d_on_lows.max():+.2f}% on "
      f"{d_on_lows.idxmax().date()}")
ranked = d_on_lows.sort_values(ascending=False)
print("     top 12 sessions by SPY closeness to its own high:")
for d, v in ranked.head(12).items():
    print(f"       {d.date()}  SPY {100*v:+.2f}% from high   XLU {C['XLU'][d]:.2f}")
for thr in (0.03, 0.05, 0.10, 0.15):
    k = int((d_on_lows >= -thr).sum())
    kk = len(novelty_t(pd.DatetimeIndex(d_on_lows[d_on_lows >= -thr].index), 21))
    print(f"     SPY within {100*thr:.0f}% of its high: {k} sessions, {kk} episodes")

# =========================================================================
# F/G. pooled: ANY SPDR at a 252d low while SPY is within 3% of its high
# =========================================================================
print("\n" + "=" * 78)
print("=== F. ANY of 11 SPDRs at a 252d closing low while SPY within 3% "
      "of its high ===")
HOR = (5, 10, 21)
pool = {h: [] for h in HOR}
pool_sp = {h: [] for h in HOR}
pool_spy = {h: [] for h in HOR}
rows = []
per_ticker = {}
ctrl_far = {h: [] for h in HOR}
ctrl_all = {h: [] for h in HOR}
for t in SECTORS:
    s = C[t]
    lo = rolling_on_valid(s, lambda x: x.rolling(252).min())
    low_mask = (s <= lo) & s.notna() & lo.notna()
    trig_raw = pd.DatetimeIndex(nyse[(low_mask & near_hi).fillna(False)])
    far_raw = pd.DatetimeIndex(nyse[(low_mask & ~near_hi).fillna(False)])
    trig = novelty_t(trig_raw, 21)
    far = novelty_t(far_raw, 21)
    per_ticker[t] = trig
    for h in HOR:
        fs = fwd_lag(s, h, 1)
        fsp = fwd_lag(C["SPY"], h, 1)
        v = fs.reindex(trig).dropna()
        pool[h].extend(v.values)
        pool_sp[h].extend((fs - fsp).reindex(trig).dropna().values)
        pool_spy[h].extend(fsp.reindex(trig).dropna().values)
        ctrl_far[h].extend(fs.reindex(far).dropna().values)
        ctrl_all[h].extend(fs.dropna().values)
        for d in trig:
            rows.append((t, d, h, fs.get(d, np.nan), (fs - fsp).get(d, np.nan)))
    print(f"   {t}: {len(trig_raw)} low-and-near-high sessions -> {len(trig)} episodes"
          f"   | {len(far)} low-and-SPY-far episodes"
          + ("   <-- includes today" if len(trig) and trig[-1] == ASOF else ""))

for h in HOR:
    print(f"\n   --- h={h} (enter MOC the session after the signal) ---")
    block(f"sector h={h}          ", pool[h])
    block(f"  ctrl low + SPY FAR  ", ctrl_far[h])
    block(f"  ctrl all sector days", ctrl_all[h])
    block(f"sector-SPY spread h={h}", pool_sp[h])
    block(f"  SPY on same anchors ", pool_spy[h])

print("\n=== G. era split of the pooled sector return ===")
df = pd.DataFrame(rows, columns=["ticker", "date", "h", "ret", "spread"])
for h in HOR:
    d = df[(df.h == h) & df.ret.notna()]
    for lab, m in (("pre-2018", d.date < pd.Timestamp(ERA)),
                   ("2018+", d.date >= pd.Timestamp(ERA))):
        w = d[m]
        if len(w):
            u, dn, n = rec(w.ret.values)
            print(f"   h={h} {lab:<9} n={n} {u}-{dn} mean {100*w.ret.mean():+.3f}% "
                  f"med {100*w.ret.median():+.3f}%")

print("\n   per-ticker record at h=10:")
d10 = df[(df.h == 10) & df.ret.notna()]
for t in SECTORS:
    w = d10[d10.ticker == t]
    if len(w):
        u, dn, n = rec(w.ret.values)
        print(f"     {t}: n={n} {u}-{dn} mean {100*w.ret.mean():+.3f}%")

print("\n   episode table (sector ret / sector-SPY spread, %):")
print("   ticker  date          h5     sp5     h10    sp10     h21    sp21")
piv = df.pivot_table(index=["ticker", "date"], columns="h",
                     values=["ret", "spread"])
for (t, d), r in piv.iterrows():
    def f(k, h):
        x = r.get((k, h), np.nan)
        return "    n/a" if (x is None or np.isnan(x)) else f"{100*x:+7.2f}"
    print(f"   {t:<6}  {d.date()}  {f('ret',5)} {f('spread',5)} "
          f"{f('ret',10)} {f('spread',10)} {f('ret',21)} {f('spread',21)}")

# =========================================================================
# H. softer XLU-only version: 21d return rank <= 3 with SPY near a high
# =========================================================================
print("\n" + "=" * 78)
print("=== H. XLU 21d return rank <= 3 while SPY is within 3% of its high ===")
r21 = pct_rank(C["XLU"], 21, 252)
print(f"   today XLU 21d rank {r21.iloc[-1]:.1f} (state says 1.6)")
mask = (r21 <= 3.0) & near_hi
raw = pd.DatetimeIndex(nyse[mask.fillna(False)])
trig = novelty_t(raw, 21)
print(f"   raw {len(raw)} sessions -> {len(trig)} episodes")
print("   " + ", ".join(str(d.date()) for d in trig))
for h in HOR:
    fs = fwd_lag(C["XLU"], h, 1)
    fsp = fwd_lag(C["SPY"], h, 1)
    print(f"   --- h={h} ---")
    block(f"XLU h={h}            ", fs.reindex(trig).dropna().values)
    block(f"  ctrl all XLU days  ", fs.dropna().values)
    block(f"XLU-SPY spread h={h} ", (fs - fsp).reindex(trig).dropna().values)
    block(f"  ctrl all days      ", (fs - fsp).dropna().values)
    v = fs.reindex(trig).dropna()
    if len(v):
        a, b = v[v.index < ERA], v[v.index >= ERA]
        print(f"     era: pre-2018 n={len(a)} mean {100*a.mean():+.3f}% | "
              f"2018+ n={len(b)} mean {100*b.mean():+.3f}%")

print("\n=== frozen levels (2026-09-18 bars) ===")
for t in ("XLU", "SPY"):
    f = raw_px[t]
    a = pd.Series(wilder_atr(f["High"], f["Low"], f["Close"]), index=f.index)
    a = a[a.index <= ASOF]
    c = f["Close"][f["Close"].index <= ASOF]
    print(f"  {t}: close {c.iloc[-1]:.4f}  Wilder-14 ATR {a.iloc[-1]:.4f} "
          f"({100*a.iloc[-1]/c.iloc[-1]:.2f}%)  bar {c.index[-1].date()}")
