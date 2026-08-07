"""E4: "Friday into the weekend at the high" -- SPY MOC Friday -> MOC Monday (1td) / +3td,
when SPY 5d-rank>=90 and within 0.5% of its 52w high.

Two specifications, both reported:
  SPEC-A (executable / the real order): trigger on day D's close, D+1 is a FRIDAY,
          enter MOC D+1, exit MOC D+1+h. This is what fires today (D=Thu 08-06).
  SPEC-B (self-referential): trigger measured on the Friday close itself, enter that
          Friday MOC. Cannot be verified for today (Friday's close does not exist yet).
Era split is decisive per the registry: famous calendar cells died post-2013.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa: F401,F403

px = load_prices(["SPY"])
spy = px["SPY"]["Close"]
idx = spy.index
rk5 = pct_rank(spy, 5)
dist = (spy / spy.rolling(252).max() - 1) * 100

# next-session weekday (the entry day), on the actual trading calendar
nxt_dow = pd.Series(idx, index=idx).shift(-1).dt.dayofweek


def fwd_entry_next(s, h):
    return s.shift(-(h + 1)) / s.shift(-1) - 1.0


def fwd_same_day(s, h):
    return s.shift(-h) / s - 1.0


def block(title, cond, fmap, valid_base, hs, min_gap_extra=1):
    print(f"\n########## {title} ##########")
    for H in hs:
        f = fmap[H]
        valid = valid_base & f.notna()
        d = idx[cond & valid]
        ep = declusters(d, H + min_gap_extra, idx)
        v = f[ep].dropna().values
        if len(v) < 3:
            print(f"h={H}: N too small ({len(v)})")
            continue
        print(f"\n--- h={H}td  day-level N={len(d)}  episodes N={len(ep)} ---")
        show([summarize(f[d].values, "TRIGGER day-level"),
              summarize(v, "TRIGGER episode-level"),
              summarize(f[valid].values, "ctrl A: SPY uncond same window"),
              summarize(f[valid & (nxt_dow == 4)].values if min_gap_extra == 1
                        else f[valid & (pd.Series(idx, index=idx).dt.dayofweek == 4)].values,
                        "ctrl C: ALL Fridays same window"),
              summarize(f[f.notna()].values, "ctrl B: all-days baseline")],
             f"h={H}")
        print(f"bootstrap P(mean<=0) LONG : {bootstrap_p_le0(v):.4f}")
        print(f"bootstrap P(mean<=0) SHORT: {bootstrap_p_le0(-v):.4f}")
        j = int(np.argmax(v)); k = int(np.argmin(v))
        print(f"best {ep[j].date()} {100*v[j]:+.2f}%  worst {ep[k].date()} {100*v[k]:+.2f}%")
        show([summarize(np.delete(v, j), "drop-BEST"), summarize(np.delete(v, k), "drop-WORST")],
             "drop-one")
        show(era_split(ep, v, "2013-01-01"), "era 2013")
        show(era_split(ep, v, "2018-01-01"), "era 2018")
        for lo, hi in [(2000, 2008), (2008, 2013), (2013, 2018), (2018, 2022), (2022, 2027)]:
            m = (ep >= pd.Timestamp(f"{lo}-01-01")) & (ep < pd.Timestamp(f"{hi}-01-01"))
            if m.sum():
                ss = summarize(v[m], f"{lo}-{hi}")
                print(f"  {ss['label']:>10s} n={ss['n']:3d} mean={ss['mean_pct']:+.3f}% "
                      f"hit={ss['hit']:.0f}% worst={ss['worst_pct']:+.2f}%")


valid_base = rk5.notna() & dist.notna()

# ---------- SPEC A: trigger Thursday, enter Friday MOC ----------
condA = (rk5 >= 90) & (dist >= -0.5) & (nxt_dow == 4)
fmapA = {h: fwd_entry_next(spy, h) for h in (1, 3, 5)}
block("E4 SPEC-A  trigger D, ENTER MOC D+1 (a Friday), exit MOC D+1+h",
      condA, fmapA, valid_base, (1, 3))

# ---------- SPEC B: trigger and entry both on the Friday close ----------
dow = pd.Series(idx, index=idx).dt.dayofweek
condB = (rk5 >= 90) & (dist >= -0.5) & (dow == 4)
fmapB = {h: fwd_same_day(spy, h) for h in (1, 3, 5)}
block("E4 SPEC-B  trigger ON the Friday close, enter that Friday MOC (self-referential)",
      condB, fmapB, valid_base, (1, 3), min_gap_extra=0)

# ---------- is the Friday condition doing anything at all? ----------
print("\n########## E4 does the FRIDAY condition add anything? (SPEC-A basis) ##########")
rows = []
for H in (1, 3):
    f = fmapA[H]
    valid = valid_base & f.notna()
    for lab, c in [("rk5>=90 + near-high, ANY weekday", (rk5 >= 90) & (dist >= -0.5)),
                   ("rk5>=90 + near-high, entry=FRI", condA),
                   ("rk5>=90 + near-high, entry=NOT FRI",
                    (rk5 >= 90) & (dist >= -0.5) & (nxt_dow != 4)),
                   ("entry=FRI only (no price cond)", nxt_dow == 4)]:
        dd = declusters(idx[c & valid], H + 1, idx)
        s = summarize(f[dd].dropna().values, f"h={H} {lab}")
        rows.append(s)
    rows.append(summarize(f[valid].values, f"h={H} -- control --"))
show(rows, "E4 Friday-condition marginal value (episodes)")

# ---------- sensitivity ----------
print("\n########## E4 sensitivity (SPEC-A, episodes) ##########")
rows = []
for H in (1, 3):
    f = fmapA[H]
    valid = valid_base & f.notna()
    for a_ in (80, 90, 95):
        for b_ in (-0.25, -0.5, -1.0):
            c = (rk5 >= a_) & (dist >= b_) & (nxt_dow == 4) & valid
            dd = declusters(idx[c], H + 1, idx)
            vv = f[dd].dropna().values
            if len(vv) < 3:
                continue
            s = summarize(vv, "")
            rows.append(dict(h=H, rk5=a_, dist=b_, n=s["n"], mean=round(s["mean_pct"], 4),
                             t=round(s["t"], 2), hit=round(s["hit"], 0),
                             worst=round(s["worst_pct"], 2)))
print(pd.DataFrame(rows).to_string(index=False))

# ---------- era detail on the widest cell, to see the fossil ----------
print("\n########## E4 the raw weekend effect on SPY by era (all Fridays, h=1) ##########")
f = fmapA[1]
valid = valid_base & f.notna()
for lo, hi in [(2000, 2008), (2008, 2013), (2013, 2018), (2018, 2022), (2022, 2027)]:
    m = valid & (nxt_dow == 4) & (idx >= pd.Timestamp(f"{lo}-01-01")) & \
        (idx < pd.Timestamp(f"{hi}-01-01"))
    ss = summarize(f[m].values, f"{lo}-{hi} all-Fri")
    mall = valid & (idx >= pd.Timestamp(f"{lo}-01-01")) & (idx < pd.Timestamp(f"{hi}-01-01"))
    sa = summarize(f[mall].values, "")
    print(f"  {ss['label']:>18s} n={ss['n']:4d} Fri->Mon mean={ss['mean_pct']:+.4f}% "
          f"| all-days 1td mean={sa['mean_pct']:+.4f}%  diff={ss['mean_pct']-sa['mean_pct']:+.4f}%")

# ---------- CPI/PPI ----------
cpi = pd.DatetimeIndex(load_events(["cpi"])["date"])
ppi = pd.DatetimeIndex(load_events(["ppi"])["date"])
both = pd.DatetimeIndex(sorted(set(cpi) | set(ppi)))
pos = pd.Series(range(len(idx)), index=idx)
for H in (1, 3):
    f = fmapA[H]
    valid = valid_base & f.notna()
    ep = declusters(idx[condA & valid], H + 1, idx)
    v = f[ep].dropna().values
    mk = []
    for d in ep:
        p = pos[d]
        if p + 1 + H >= len(idx):
            mk.append(False); continue
        lo, hi = idx[p + 1], idx[p + 1 + H]
        mk.append(bool(((both > lo) & (both <= hi)).any()))
    mk = np.array(mk, dtype=bool)
    show([summarize(v[mk], "CPI/PPI in hold"), summarize(v[~mk], "neither")],
         f"E4 h={H} CPI/PPI split ({mk.sum()}/{len(mk)})")
