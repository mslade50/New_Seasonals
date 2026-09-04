"""^VIX3M at a 52-week low on the eve of the symposium.

Tonight ^VIX3M closed 17.56, exactly its trailing-252 minimum, ^VIX 14.51
(21d rank 6.0) and ^VVIX 82.9 (21d rank 4.0). The engine's P2/P2b forward
means are weak and do not pass BH, so the question is not "what happens next"
but how unusual the STATE is, how broad the compression is, and whether the
one-year floor has ever coincided with the symposium before.

Anchor = the session the state printed, so h1 is the next session.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_events, fwd_ret, summarize, show, sign_test,
    era_split, cluster_note, declusters, rolling_on_valid,
)

TK = ["^VIX", "^VIX3M", "^VVIX", "SPY"]
px = close_panel(TK)
dates = px.index

v3 = px["^VIX3M"].dropna()
print(f"^VIX3M history {v3.index.min().date()} .. {v3.index.max().date()}, "
      f"n={len(v3)}, last {v3.iloc[-1]:.2f}")

# 52-week low state, on the ticker's OWN valid calendar (rolling_on_valid)
lo252 = rolling_on_valid(px["^VIX3M"], lambda x: x.rolling(252).min())
at_low = (px["^VIX3M"] <= lo252 * 1.0000001) & lo252.notna()
low_days = dates[at_low.fillna(False)]
print(f"^VIX3M sessions at a trailing-252 low: {len(low_days)}")

# first such low in 30+ / 90+ calendar days = the engine's P2 / P2b
def first_in_days(idx, gap_days):
    keep, last = [], None
    for d in sorted(idx):
        if last is None or (d - last).days >= gap_days:
            keep.append(d)
        last = d
    return pd.DatetimeIndex(keep)

p2 = first_in_days(low_days, 30)
p2b = first_in_days(low_days, 90)
print(f"P2 (first in 30+ cal days): {len(p2)}  |  P2b (90+): {len(p2b)}")
print("P2b episode dates:", [str(d.date()) for d in p2b])

# --- how rare is the whole-complex compression, not just VIX3M ---
def rank21(s):
    r21 = s / s.shift(21) - 1.0
    return rolling_on_valid(r21, lambda x: x.rolling(252).rank(pct=True)) * 100

r_vix, r_v3, r_vvix = rank21(px["^VIX"]), rank21(px["^VIX3M"]), rank21(px["^VVIX"])
triple = (r_vix <= 10) & (r_v3 <= 10) & (r_vvix <= 10)
triple_days = dates[triple.fillna(False)]
print(f"\nsessions with VIX, VIX3M and VVIX 21d-return rank all <= 10: "
      f"{len(triple_days)} of {int((r_vix.notna() & r_v3.notna() & r_vvix.notna()).sum())} "
      f"with all three defined")
if len(triple_days):
    print("  years:", pd.Series(triple_days.year).value_counts().sort_index().to_dict())

# --- has a VIX3M 52w low ever landed near a symposium? ---
jh = pd.DatetimeIndex(load_events(["jackson_hole"])["date"])
pos = pd.Series(range(len(dates)), index=dates)
print(f"\nVIX3M 52w-low sessions within 5 td of a symposium:")
hits = 0
for d in low_days:
    p = pos.get(d)
    for j in jh:
        pj = pos.get(j)
        if pj is None:
            prior = dates[dates < j]
            pj = pos.get(prior[-1]) if len(prior) else None
        if pj is not None and abs(p - pj) <= 5:
            print(f"  {d.date()} (symposium {j.date()}, {p-pj:+d} td)")
            hits += 1
            break
print(f"  total: {hits}")

# --- forward behaviour, stated honestly ---
for name, idx in [("P2 first-low-in-30d", p2), ("P2b first-low-in-90d", p2b)]:
    print(f"\n{'='*68}\n{name}  n={len(idx)}\n{'='*68}")
    for sub in ["^VIX3M", "^VIX", "SPY"]:
        rows = []
        for h in (1, 5, 21):
            v = fwd_ret(px[sub], h).reindex(idx).dropna()
            rows.append(summarize(v.values, f"{sub} h{h}"))
        show(rows, "")
        v1 = fwd_ret(px[sub], 1).reindex(idx).dropna()
        k = int((v1.values > 0).sum())
        print(f"    h1 record {k}-{len(v1)-k} up, sign p {sign_test(k, len(v1)):.4f}")
        print(f"    h1 era: {[(r['label'], r['n'], round(r['mean_pct'],3)) for r in era_split(v1.index, v1.values)]}")
        print(f"    h1 concentration: {cluster_note(v1.index, v1.values)}")
        base = fwd_ret(px[sub], 1).dropna()
        print(f"    h1 all-days baseline: {100*base.mean():+.3f}% "
              f"hit {100*(base.values>0).mean():.1f}% n={len(base)}")

# --- ^VVIX 21d bottom-5pct cell: mean vs median, the tail question ---
print(f"\n{'='*68}\n^VVIX 21d return in the bottom 5% of its year\n{'='*68}")
r = rank21(px["^VVIX"])
m = (r <= 5).fillna(False)
idx = dates[m]
v1 = fwd_ret(px["^VVIX"], 1).reindex(idx).dropna()
s = summarize(v1.values, "^VVIX h1")
k = int((v1.values > 0).sum())
print(f"  n={s['n']} mean {s['mean_pct']:+.3f}% median {s['median_pct']:+.3f}% "
      f"hit {s['hit']:.1f}% t {s['t']:+.2f} record {k}-{s['n']-k} up "
      f"sign p {sign_test(k, s['n']):.4f}")
dc = declusters(idx, 10, dates)
vd = fwd_ret(px["^VVIX"], 1).reindex(dc).dropna()
sd_ = summarize(vd.values, "declustered")
kd = int((vd.values > 0).sum())
print(f"  declustered (10td) n={sd_['n']} mean {sd_['mean_pct']:+.3f}% "
      f"median {sd_['median_pct']:+.3f}% record {kd}-{sd_['n']-kd} up")
print(f"  {cluster_note(v1.index, v1.values)}")

# where SPY sits while the vol complex is on the floor
print(f"\nSPY 21d rank tonight vs the VIX3M-low sessions:")
spy_r21 = rolling_on_valid(px["SPY"] / px["SPY"].shift(21) - 1.0,
                           lambda x: x.rolling(252).rank(pct=True)) * 100
sel = spy_r21.reindex(low_days).dropna()
print(f"  n={len(sel)} median SPY 21d rank on VIX3M 52w-low days: {sel.median():.1f}")
print(f"  share with SPY 21d rank >= 90: {100*(sel>=90).mean():.1f}%  (tonight: 91.3)")
