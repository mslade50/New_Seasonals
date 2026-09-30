"""Posts check (2026-09-25) part 2: G. breadth divergence.

Universe = data/posts_state.json tape.universe (TODAY's ~218 names applied to
all history: SURVIVORSHIP BIAS, names that died or were dropped are absent).
Per session: share of names whose 21-session return pct_rank (252 lookback,
pitch_lab.pct_rank, the posts_state convention) is above 50, counted only on
sessions with >= 150 names ranked. Trigger: share <= 20% AND SPY close within
1% of its 252-session closing high (min_periods 200). Declustered 21.
Forward SPY from the D+1 open at h10 / h21 (Open[D+1] -> Close[D+1+h]), plus
lag-0 reference and the +/-126 local control. Same helpers as 01_cells.py.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import (  # noqa: E402
    cluster_note, declusters, load_prices, local_control, pct_rank, sign_test,
    summarize,
)

ASOF = pd.Timestamp("2026-09-25")
ERA = pd.Timestamp("2018-01-01")
MIN_NAMES = 150

t0 = time.time()
state = json.loads((ROOT / "data" / "posts_state.json").read_text())
uni = sorted(state["tape"]["universe"].keys())
print(f"universe from posts_state tape.universe: {len(uni)} names | posts_state "
      f"breadth pct_rank21_above_50 = {state['tape']['breadth'].get('pct_rank21_above_50')}")
raw = load_prices(uni + ["SPY"])
print(f"loaded {len(raw)} frames in {time.time()-t0:.1f}s")

nyse = raw["SPY"].index
nyse = pd.DatetimeIndex(nyse[nyse <= ASOF])
SPY = raw["SPY"][raw["SPY"].index <= ASOF].astype(float).reindex(nyse)

ranks = {}
for t in uni:
    if t not in raw:
        continue
    c = raw[t]["Close"]
    c = c[c.index <= ASOF].astype(float).dropna()
    ranks[t] = pct_rank(c, 21, 252).reindex(nyse)
R = pd.DataFrame(ranks)
valid = R.notna().sum(axis=1)
share = ((R > 50).sum(axis=1) / valid * 100).where(valid >= MIN_NAMES)
print(f"ranked {R.shape[1]} names in {time.time()-t0:.1f}s | first session with >= "
      f"{MIN_NAMES} names ranked: {share.first_valid_index().date()}")
print("  valid-name count by year start: " + ", ".join(
    f"{y}:{int(valid[valid.index.year == y].iloc[0])}" for y in
    (2000, 2002, 2004, 2005, 2006, 2008, 2010, 2015, 2020, 2026)))
print(f"  TONIGHT share = {share.iloc[-1]:.1f}% ({int(valid.iloc[-1])} ranked)")
print("  last 6 sessions: " + ", ".join(f"{d.date()} {v:.1f}" for d, v in share.tail(6).items()))

spyc = SPY["Close"]
spy_dh = spyc / spyc.rolling(252, min_periods=200).max() - 1.0
print(f"  SPY vs 252d closing high {100*spy_dh.iloc[-1]:+.2f}%")


def o2c(h: int) -> pd.Series:
    return spyc.shift(-(1 + h)) / SPY["Open"].shift(-1) - 1.0


def c2c0(h: int) -> pd.Series:
    return spyc.shift(-h) / spyc - 1.0


def rec(v) -> tuple[int, int, int]:
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return int((v > 0).sum()), int((v < 0).sum()), len(v)


def era_str(v: pd.Series) -> str:
    out = []
    for lab, x in (("pre-2018", v[v.index < ERA]), ("2018+", v[v.index >= ERA])):
        if len(x) == 0:
            out.append(f"{lab} n=0")
            continue
        u, d, n = rec(x.values)
        out.append(f"{lab} n={n} {u}-{d} mean {100*x.mean():+.2f}%")
    return " | ".join(out)


def best2(v: pd.Series) -> str:
    s = v.sort_values(ascending=False)
    tot = float(v.sum())
    top = float(s.iloc[:2].sum())
    share_ = 100 * top / tot if tot != 0 else np.nan
    ex = float(s.iloc[2:].mean()) if len(s) > 2 else np.nan
    return (f"best2 {[str(i.date()) for i in s.index[:2]]} {100*top:+.2f}pp of "
            f"{100*tot:+.2f}pp total ({share_:.0f}%) | mean ex-best2 {100*ex:+.3f}% | "
            f"worst {v.idxmin().date()} {100*v.min():+.2f}%")


def block(label: str, s: pd.Series, trig: pd.DatetimeIndex, start: str) -> None:
    v = s.reindex(trig).dropna()
    v = v[v.index >= start]
    allv = s[s.index >= start].dropna()
    if len(v) == 0:
        print(f"  {label}: n=0")
        return
    u, d, n = rec(v.values)
    sm = summarize(v.values)
    base_hit = float((allv > 0).mean())
    loc = local_control(nyse, pd.DatetimeIndex(v.index), 126)
    locv = s.reindex(loc).dropna()
    lu, ld, _ = rec(locv.values)
    print(f"  {label}: n={n} {u}-{d} mean {sm['mean_pct']:+.3f}% med "
          f"{sm['median_pct']:+.3f}% t {sm['t']:+.2f} | ctrl all {100*allv.mean():+.3f}% "
          f"(hit {100*base_hit:.1f}%) local+/-126 {100*locv.mean():+.3f}% ({lu}-{ld}) | "
          f"sign p(up) {sign_test(u, n):.4f} vs base-rate {sign_test(u, n, base_hit):.4f}")
    print(f"      era: {era_str(v)}")
    print(f"      conc: {best2(v)}")
    print(f"      cluster_note: {cluster_note(v.index, v.values)}")
    if n <= 30:
        print("      dates: " + ", ".join(f"{i.date()} {100*x:+.2f}" for i, x in v.items()))


start = str(share.first_valid_index().date())
for nm, mk in (("share <= 20 AND SPY within 1% of 252d high", (share <= 20) & (spy_dh >= -0.01)),
               ("share <= 25 AND SPY within 1% of 252d high", (share <= 25) & (spy_dh >= -0.01)),
               ("reference: share <= 20, any SPY level", (share <= 20))):
    m = mk.eq(True)
    d = nyse[m.to_numpy()]
    epi_all = declusters(d, 21, nyse)
    print(f"\n  ##### {nm}: raw {len(d)} -> declustered(21) {len(epi_all)} | latest starts "
          f"{[str(x.date()) for x in epi_all[-3:]]} | tonight starts new episode "
          f"{ASOF in epi_all} #####")
    epi = epi_all[epi_all < ASOF]
    for h in (10, 21):
        block(f"G SPY h{h} OPEN D+1 [{nm}]", o2c(h), epi, start)
    for h in (10, 21):
        v = c2c0(h).reindex(epi).dropna()
        u, dd, n = rec(v.values)
        print(f"    SPY h{h} CLOSE D lag0 ref: n={n} {u}-{dd} mean {100*v.mean():+.3f}%")
print(f"\nSURVIVORSHIP: universe is today's posts_state list applied to all history.")
print(f"total runtime {time.time()-t0:.1f}s\ndone.")
