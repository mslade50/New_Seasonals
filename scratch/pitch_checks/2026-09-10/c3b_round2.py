"""c3b — ROUND 2 on C3. The three questions round 1 raised.

Round 1 said: DENSE pays in EVERY compression bucket (+0.18 to +3.36pp over
clear across all six), the live cell has no SPY residual (beta 1.83, alpha
+0.469% t +0.36), and the -0.5x era cell reads N=11 / +1.419% / 81.8% / sign
p 0.033 at h=6.

So round 2 asks the only three things that can still kill or save it:

  Q1  GATE ATTRIBUTION WITHOUT THE GATE. What does DENSE alone do, with no
      compression condition at all, on SVXY? If DENSE-alone >= DENSE x
      compression, the compression gate is a filter that does not filter and
      the candidate's entire stated reason for existing is void.

  Q2  IS IT THE EQUITY LEG OR THE ERA? Measure the IDENTICAL dense-calendar
      cell on LONG SPY over (a) the full 26-year sample and (b) the SVXY era
      only. C1 already measured (a) at +0.078% t 0.44 on 254 episodes. If
      SPY's dense cell is much richer inside the SVXY window, C3 is era
      selection on the equity leg, levered ~1.8x.

  Q3  DECLUSTER ORDER (rule 9, non-commutative) and episode-level
      concentration in the -0.5x era, where N=11.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

EVENT_KINDS = ["nfp", "cpi", "ppi", "fomc_decision", "opex",
               "quad_witching", "vix_expiry"]
WIN, THRESH = 6, 4
RELEVER = pd.Timestamp("2018-02-28")
SVXY_START = pd.Timestamp("2011-10-04")

px = close_panel(["SPY", "^VIX", "SVXY"])
spy = px["SPY"].dropna()
idx = spy.index
vix = px["^VIX"].reindex(idx).ffill()
pos = pd.Series(range(len(idx)), index=idx)
ev = load_events(EVENT_KINDS)

locs = idx.searchsorted(pd.DatetimeIndex(ev["date"]))
cnt = np.zeros(len(idx))
for L in locs[locs < len(idx)]:
    cnt[L] += 1
cs = np.concatenate([[0.0], np.cumsum(cnt)])
dens = pd.Series([np.nan if p + WIN >= len(idx) else cs[p + WIN + 1] - cs[p + 1]
                  for p in range(len(idx))], index=idx)
dense = (dens >= THRESH).fillna(False)

rr = rolling_on_valid(vix, lambda x: (x.rolling(21).max() - x.rolling(21).min())
                      / x.rolling(21).mean())
rrp = rolling_on_valid(rr, lambda x: x.rolling(252).rank(pct=True) * 100.0)
comp_lo = (rrp > 0) & (rrp <= 5)
comp_mid = (rrp > 5) & (rrp <= 15)
comp_any = (rrp > 0) & (rrp <= 15)


def cell(ret, valid, mask, h, label, floor=None):
    t = pd.DatetimeIndex(idx[mask.fillna(False).values]).intersection(valid)
    if floor is not None:
        t = t[t >= floor]
    if len(t) == 0:
        return {"label": label, "n": 0}
    e = declusters(t, max(h, WIN), valid)
    v = ret.loc[e].values
    s = summarize(v, label)
    s["n_days"] = len(t)
    s["signp"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    return s


# ------------------------------------------------------------------- Q1
print("=" * 80)
print("Q1. GATE ATTRIBUTION WITHOUT THE GATE — LONG SVXY")
print("    Does the compression condition add ANYTHING over DENSE alone?")
for H in (5, 6, 10):
    ret = vehicle_ret(px, [("SVXY", 1.0)], H, 1)
    valid = ret.dropna().index
    rows = [
        cell(ret, valid, dense, H, "DENSE alone (no compression gate)"),
        cell(ret, valid, dense & comp_any, H, "DENSE x compression<=15 (parent)"),
        cell(ret, valid, dense & comp_mid, H, "DENSE x (5,15]  <- TODAY'S CELL"),
        cell(ret, valid, dense & comp_lo, H, "DENSE x (0,5]   <- PITCHED CELL"),
        cell(ret, valid, dense & ~comp_any.fillna(False), H,
             "DENSE x compression>15 (DISCARDED COMPLEMENT)"),
        cell(ret, valid, ~dense & dens.notna(), H, "clear alone"),
        cell(ret, valid, pd.Series(True, index=idx), H, "all SVXY days"),
    ]
    show(rows, f"   LONG SVXY, h={H}, episodes (decluster gap {max(H, WIN)})")
    d_alone = [r for r in rows if r["label"].startswith("DENSE alone")][0]
    d_mid = [r for r in rows if "TODAY" in r["label"]][0]
    d_lo = [r for r in rows if "PITCHED" in r["label"]][0]
    d_hi = [r for r in rows if "COMPLEMENT" in r["label"]][0]
    print(f"   -> compression (5,15] adds {d_mid['mean_pct']-d_alone['mean_pct']:+.3f}pp "
          f"over DENSE alone;  (0,5] adds "
          f"{d_lo['mean_pct']-d_alone['mean_pct']:+.3f}pp;  the DISCARDED "
          f"complement (>15) is {d_hi['mean_pct']:+.3f}% on N={d_hi['n']}")

# ------------------------------------------------------------------- Q2
print("\n" + "=" * 80)
print("Q2. IS IT THE EQUITY LEG OR THE ERA? Identical cells on LONG SPY.")
for H in (5, 6, 10):
    ret = vehicle_ret(px, [("SPY", 1.0)], H, 1)
    valid = ret.dropna().index
    rows = [
        cell(ret, valid, dense, H, "DENSE, FULL 26y sample"),
        cell(ret, valid, dense, H, "DENSE, SVXY era 2011-10+", floor=SVXY_START),
        cell(ret, valid, dense, H, "DENSE, -0.5x era 2018-02+", floor=RELEVER),
        cell(ret, valid, dense & comp_mid, H, "DENSE x (5,15], full"),
        cell(ret, valid, dense & comp_mid, H, "DENSE x (5,15], -0.5x era",
             floor=RELEVER),
        cell(ret, valid, pd.Series(True, index=idx), H, "all days, FULL"),
        cell(ret, valid, pd.Series(True, index=idx), H, "all days, SVXY era",
             floor=SVXY_START),
        cell(ret, valid, pd.Series(True, index=idx), H, "all days, -0.5x era",
             floor=RELEVER),
    ]
    show(rows, f"   LONG SPY, h={H}, episodes")

print("\n   The arithmetic that decides it (h=6): SVXY's live cell was")
print("   +1.483% with beta 1.83 on SPY. If SPY's own dense cell in the SVXY")
print("   era is ~+0.8%, then 1.83 x 0.8 = ~1.5% and the vol vehicle is")
print("   contributing nothing but leverage.")

# ------------------------------------------------------------------- Q3
print("\n" + "=" * 80)
print("Q3. DECLUSTER ORDER (rule 9) + concentration, -0.5x era, h=6")
H = 6
ret = vehicle_ret(px, [("SVXY", 1.0)], H, 1)
valid = ret.dropna().index
valid5 = valid[valid >= RELEVER]

m = (dense & comp_mid).fillna(False)
t_raw = pd.DatetimeIndex(idx[m.values]).intersection(valid5)

# order A: FILTER then DECLUSTER (what round 1 did)
eA = declusters(t_raw, max(H, WIN), valid5)
# order B: DECLUSTER the dense mask first, THEN apply the compression filter
t_dense = pd.DatetimeIndex(idx[dense.values]).intersection(valid5)
eB0 = declusters(t_dense, max(H, WIN), valid5)
eB = pd.DatetimeIndex([d for d in eB0 if bool(comp_mid.get(d, False))])
for lbl, e in (("A: filter-then-decluster", eA), ("B: decluster-then-filter", eB)):
    v = ret.loc[e].values
    s = summarize(v, lbl)
    s["signp"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    show([s], f"   {lbl}")
    print(f"     dates: {', '.join(str(d.date()) for d in e)}")
    print(f"     concentration: {cluster_note(e, v)}")

v = ret.loc[eA].values
o = np.argsort(-v)
print(f"\n   drop-best-2 (order A): {100*np.delete(v, o[:2]).mean():+.3f}% "
      f"on N={len(v)-2}")
yrs = pd.DatetimeIndex(eA).year
by_y = pd.Series(v).groupby(yrs.values).sum()
print(f"   per-year contribution: "
      f"{dict((int(y), round(100*r, 2)) for y, r in by_y.items())}")
keep = yrs != by_y.idxmax()
print(f"   drop-best-year ({by_y.idxmax()}): {100*v[keep].mean():+.3f}% "
      f"on N={int(keep.sum())}")

# how many DISTINCT calendar episodes are these really?
print(f"\n   episodes are {len(eA)}; distinct (year, month) pairs = "
      f"{len(set((d.year, d.month) for d in eA))} -> "
      f"{sorted(set((d.year, d.month) for d in eA))}")

# ------------------------------------------------------------------ live-cell
print("\n" + "=" * 80)
print("BOTTOM LINE INPUTS")
print(f"   today's rel-range pctile = {rrp.dropna().iloc[-1]:.3f} "
      f"(pre-specified trigger <= 5 -> NOT LIVE)")
print("   today's cell is (5,15] x DENSE, which the candidate did not pitch.")
