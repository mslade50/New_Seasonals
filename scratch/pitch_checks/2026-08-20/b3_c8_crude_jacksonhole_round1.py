"""C8 round 1: long crude through Jackson Hole, entry 6 sessions before the
conference (today's slot). Energy is the one asset class the JH sweep never
reached; the registry's prior (2026-08-13, -08-11, -08-18) is that the anchor
is decoration on a late-August seasonal, and the ladder is 9-for-9.

Vehicles compared as WHOLE variants: USO (roll decay, registry-established),
XLE (crude beta 0.479 per registry 2026-08-11) and CL=F front continuous.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
warnings.filterwarnings("ignore")
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

TK = ["USO", "XLE", "CL=F", "XOP", "SPY"]
px = close_panel(TK)
d = px.index

ev = load_events(["jackson_hole"])
jh_raw = pd.DatetimeIndex(sorted(ev["date"]))
print("Jackson Hole dates in macro_events.csv:")
print("  " + ", ".join(f"{x.date()}" for x in jh_raw))
jh = pd.DatetimeIndex(sorted(set(jh_raw) & set(d)))
print(f"JH dates aligned to the price index: {len(jh)} of {len(jh_raw)}")
# an event date past the end of the index must never mint a fake anchor
jh = jh[jh <= d[-1]]

OFF = -6


def anchor_at(off: int) -> pd.DatetimeIndex:
    pos = d.get_indexer(jh) + off
    pos = pos[(pos >= 0) & (pos < len(d))]
    return d[pos]


entry = anchor_at(OFF)
print(f"\nentry anchors (JH{OFF}) N={len(entry)}: "
      f"{', '.join(str(x.date()) for x in entry)}")
tdom = []
for a in entry:
    m = d[(d.year == a.year) & (d.month == a.month)]
    tdom.append(int(np.where(m == a)[0][0]) + 1)
print(f"anchor trading-day-of-month {min(tdom)}..{max(tdom)} "
      f"(median {int(np.median(tdom))}); today is Aug tdom 14")
for t in TK:
    s = px[t].dropna()
    print(f"  {t:6s} first bar {s.index[0].date()}  usable anchors "
          f"{len(fwd_lag(px[t], 10, lag=0).reindex(entry).dropna())}")

mask = pd.Series(False, index=d)
mask.loc[anchor_at(OFF - 1)] = True   # lag=1 -> entry at the JH-6 close

# ---------------------------------------------------------------- round 1
for h in (5, 6, 10):
    battery(px, mask, [("USO", 1.0)], h, "C8 LONG USO, JH-6 entry",
            cost_bps=8.0, min_gap=15, event_kinds=("fomc_decision",))

# ---------------------------------------------------- offset placebo ladder
print("\n\n" + "=" * 78)
print("OFFSET PLACEBO LADDER — k=-10..+5 sessions from Jackson Hole")
print("=" * 78)
for tkr in ("USO", "CL=F", "XLE"):
    for h in (6, 10):
        rows = []
        for k in range(-10, 6):
            a = anchor_at(k)
            s = fwd_lag(px[tkr], h, lag=0).reindex(a).dropna()
            if len(s) < 5:
                continue
            rows.append({"k": k, "n": len(s), "mean_pct": 100 * s.mean(),
                         "hit": 100 * (s > 0).mean(),
                         "true": "<== TRUE" if k == OFF else ""})
        df = pd.DataFrame(rows).sort_values("mean_pct", ascending=False)
        df["rank"] = range(1, len(df) + 1)
        print(f"\n--- {tkr} h={h} ---")
        print(df.round(3).to_string(index=False))
        tr = df[df["k"] == OFF]
        print(f"  >>> TRUE ANCHOR (k={OFF}) RANKS {int(tr['rank'].iloc[0])} "
              f"of {len(df)} at h={h}")

# --------------------------------------- unconditional late-August window
print("\n\n" + "=" * 78)
print("UNCONDITIONAL LATE-AUGUST WINDOW — the control that closed JH on large "
      "caps (tdom band matched to the anchor, no event involved)")
print("=" * 78)
aug = d[d.month == 8]
tmap = {}
for y in sorted(set(aug.year)):
    m = d[(d.year == y) & (d.month == 8)]
    for i, x in enumerate(m):
        tmap[x] = i + 1
lo, hi = min(tdom), max(tdom)
band = pd.DatetimeIndex([x for x in aug if lo <= tmap[x] <= hi])
band616 = pd.DatetimeIndex([x for x in aug if 6 <= tmap[x] <= 16])
for tkr in ("USO", "CL=F", "XLE"):
    print(f"\n{tkr}:")
    for h in (5, 6, 10):
        rj = fwd_lag(px[tkr], h, lag=0).reindex(entry).dropna()
        rb = fwd_lag(px[tkr], h, lag=0).reindex(band).dropna()
        r616 = fwd_lag(px[tkr], h, lag=0).reindex(band616).dropna()
        rall = fwd_lag(px[tkr], h, lag=0).dropna()
        print(f"  h={h:2d}  JH anchor {100*rj.mean():+.3f}% (N={len(rj)}, "
              f"hit {100*(rj>0).mean():.0f}%)  |  Aug tdom {lo}-{hi} "
              f"{100*rb.mean():+.3f}% (N={len(rb)})  |  Aug tdom 6-16 "
              f"{100*r616.mean():+.3f}% (N={len(r616)})  |  all-days "
              f"{100*rall.mean():+.3f}%  ->  anchor minus Aug window "
              f"{100*(rj.mean()-rb.mean()):+.3f}pp")

# ------------------------------------------------------------ midterm split
print("\n\n" + "=" * 78)
print("MIDTERM SPLIT. 2026 IS MIDTERM.")
print("=" * 78)
for tkr in ("USO", "CL=F", "XLE"):
    for h in (6, 10):
        s = fwd_lag(px[tkr], h, lag=0).reindex(entry).dropna()
        mid = s.index.year % 4 == 2
        print(f"{tkr:6s} h={h:2d}: MIDTERM {100*s[mid].mean():+.3f}% "
              f"N={int(mid.sum())} {sorted(set(s.index[mid].year))} "
              f"(hit {100*(s[mid]>0).mean():.0f}%)  |  non-midterm "
              f"{100*s[~mid].mean():+.3f}% N={int((~mid).sum())}")

# ------------------------------------------------------ concentration/years
print("\n\n" + "=" * 78)
print("CONCENTRATION: by-year and drop-best-k")
print("=" * 78)
for tkr in ("USO", "CL=F"):
    for h in (6, 10):
        s = fwd_lag(px[tkr], h, lag=0).reindex(entry).dropna()
        by = pd.Series((100 * s.values).round(2),
                       index=s.index.year).sort_values(ascending=False)
        v = np.sort(s.values)[::-1]
        print(f"\n--- {tkr} h={h} (N={len(s)}) ---")
        print(by.to_string())
        print(f"  full {100*v.mean():+.3f}%  drop-1 {100*v[1:].mean():+.3f}%  "
              f"drop-2 {100*v[2:].mean():+.3f}%  drop-3 {100*v[3:].mean():+.3f}%")
        print(f"  {cluster_note(s.index, s.values, k=2)}")
        wins = int((s > 0).sum())
        print(f"  record {wins}-{len(s)-wins}, sign p "
              f"{sign_test(wins, len(s)):.4f}, bootstrap P(mean<=0) "
              f"{bootstrap_p_le0(s.values):.3f}")

# ------------------------------------------------------------------ vehicle
print("\n\n" + "=" * 78)
print("VEHICLE AS WHOLE VARIANTS, and the roll question")
print("=" * 78)
print("Round-trip cost assumptions: USO ~8 bps (spread+slip), XLE ~4 bps, "
      "CL=F ~4 bps (1 tick on a $60 barrel = 1.7 bps each way).")
for h in (5, 6, 8, 10):
    row = []
    for tkr, c in (("USO", 8.0), ("XLE", 4.0), ("CL=F", 4.0), ("XOP", 10.0)):
        s = fwd_lag(px[tkr], h, lag=0).reindex(entry).dropna()
        row.append(f"{tkr} {100*s.mean():+6.2f}% N={len(s):2d} "
                   f"({100*s.mean()*100/c:.1f}x cost)")
    print(f"  h={h:2d}  " + " | ".join(row))

print("\n--- USO vs CL=F on the SAME anchors: is the long the mirror image of "
      "USO's conditional roll cost? ---")
for h in (5, 6, 10):
    u = fwd_lag(px["USO"], h, lag=0).reindex(entry).dropna()
    c = fwd_lag(px["CL=F"], h, lag=0).reindex(entry).dropna()
    j = u.index.intersection(c.index)
    print(f"  h={h:2d}  USO {100*u.loc[j].mean():+.3f}%  CL=F "
          f"{100*c.loc[j].mean():+.3f}%  USO-minus-CL "
          f"{100*(u.loc[j]-c.loc[j]).mean():+.3f}pp (N={len(j)})")
    ua = fwd_lag(px["USO"], h, lag=0).dropna()
    ca = fwd_lag(px["CL=F"], h, lag=0).dropna()
    jj = ua.index.intersection(ca.index)
    print(f"        all-days USO-minus-CL {100*(ua.loc[jj]-ca.loc[jj]).mean():+.3f}pp "
          f"(the unconditional roll drag)")
