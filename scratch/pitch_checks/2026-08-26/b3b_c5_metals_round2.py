"""b3b / C5 round 2.

Round 1 killed the LONG (count>=4 pays -0.352% GDX / -0.469% GLD against
all-days controls of +0.272% / +0.216%). "Direction from the data" then
points at the FADE, so the fade is what round 2 has to close.

Three tests:
 1. GATE ATTRIBUTION against the CLOSED single-name cell. GDX is itself a
    member of the complex; if count>=4 days are almost all GDX-rank>=95 days,
    the "breadth count" is the 2026-08-17/25 GDX thrust cell with a
    decoration gate bolted on, and it inherits that closure.
 2. COMPOSITION. Today is 3 equity + 1 metal. Round 1 already showed that
    slice pays +0.897% while the headline count cell pays -0.352%. Enumerate
    the compositions and show the answer is manufactured by the slicing.
 3. CONCENTRATION / drop-best on the fade, plus era, midterm, and the
    definition neighbours (rank threshold 90/95/98, lookback 21/10/42).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change
import pandas as pd, numpy as np

pd.set_option("display.width", 240)

CORE6 = ["GLD", "SLV", "GDX", "NEM", "FCX", "XME"]
EQ = ["GDX", "NEM", "FCX", "XME"]
MET = ["GLD", "SLV"]
allt = sorted(set(CORE6 + ["SPY"]))
raw = load_prices(allt)
cal = raw["GDX"]["Close"].dropna().index
px = pd.DataFrame({t: raw[t]["Close"] for t in allt}).reindex(cal)
px = px.dropna(subset=["GDX", "GLD"])
cal = px.index


def rk(t, n=21, lb=252):
    return pct_rank(raw[t]["Close"].dropna(), n, lb).reindex(cal)


def cnt(members, n=21, lb=252, thr=95.0):
    m = pd.DataFrame({t: (rk(t, n, lb) >= thr).fillna(False) for t in members})
    ok = pd.DataFrame({t: rk(t, n, lb).notna() for t in members})
    return m.sum(axis=1).where(ok.all(axis=1))


c6 = cnt(CORE6)
ret = vehicle_ret(px, [("GDX", 1.0)], 5, 1)
valid = ret.dropna().index
retg = vehicle_ret(px, [("GLD", 1.0)], 5, 1)


def cell(mask, r=ret, v=valid, lbl="", h=5):
    tt = cal[mask.reindex(cal, fill_value=False).values].intersection(v)
    epi = declusters(tt, h, v)
    d = summarize(r.loc[epi].values, lbl)
    d["n_days"] = len(tt)
    return d, epi


# ---------- 1. GATE ATTRIBUTION vs the closed single-name GDX thrust ----------
print("=" * 78)
print("1. GATE ATTRIBUTION -- is this the CLOSED GDX single-name thrust in disguise?")
print("=" * 78)
gdx95 = rk("GDX") >= 95
live = c6 >= 4
ov = (live & gdx95).sum() / max(1, live.sum())
print(f"  count>=4 days that are ALSO GDX rank21>=95 days: {int((live & gdx95).sum())} of "
      f"{int(live.sum())} = {100*ov:.1f}%")
print(f"  GDX rank21>=95 days that are NOT count>=4: {int((gdx95 & ~live).sum())} of "
      f"{int(gdx95.sum())}")
rows = []
for lbl, m in [("ALL DAYS", pd.Series(True, index=cal)),
               ("GDX rank21>=95 ALONE (the closed cell)", gdx95),
               ("CORE6 count>=4 (pitched)", live),
               ("GDX>=95 & count>=4 (joint)", gdx95 & live),
               ("GDX>=95 & count<4 (thrust WITHOUT breadth)", gdx95 & (c6 < 4)),
               ("count>=4 & GDX<95 (breadth WITHOUT GDX)", live & ~gdx95)]:
    d, _ = cell(m, lbl=lbl)
    rows.append(d)
show(rows, "LONG GDX h=5, episodes")
print("  A gate that discards observations to move the mean by <0.1pp is decoration.")

# ---------- 2. COMPOSITION ----------
print("\n" + "=" * 78)
print("2. COMPOSITION -- today is 3 equity + 1 metal. Does the count mean one thing?")
print("=" * 78)
ceq = cnt(EQ)
cmet = cnt(MET)
print(f"  TODAY equity={ceq.iloc[-1]:.0f}/4  metal={cmet.iloc[-1]:.0f}/2  total={c6.iloc[-1]:.0f}/6")
rows = []
for e in range(0, 5):
    for m in range(0, 3):
        if e + m < 3:
            continue
        d, _ = cell((ceq == e) & (cmet == m), lbl=f"equity={e}, metal={m}")
        if d["n"] >= 3:
            rows.append(d)
rows.append(cell(pd.Series(True, index=cal), lbl="ALL DAYS")[0])
show(rows, "LONG GDX h=5 by exact composition (episodes)")
print("  Three defensible readings of TODAY's state give three different signs;")
print("  see also round 1: count==4 -0.861%, equity>=3&metal>=1 +0.897%, count==6 +2.911%.")

# ---------- 3. FADE: concentration, drop-best, splits, definition neighbours ----------
print("\n" + "=" * 78)
print("3. THE FADE (short GDX at count>=4) -- concentration and definition stability")
print("=" * 78)
d, epi = cell(live, lbl="count>=4")
f = -ret.loc[epi].values                     # fade = short
base = -float(ret.loc[valid].mean())
print(f"  SHORT GDX h=5 on count>=4: mean {100*f.mean():+.3f}%  N={len(f)}  "
      f"hit {100*(f>0).mean():.1f}%  vs all-days short control {100*base:+.3f}% "
      f"-> excess {100*(f.mean()-base):+.3f}pp")
print(f"  bootstrap P(mean<=0) = {bootstrap_p_le0(f):.3f}   "
      f"sign p (own base {100*(-ret.loc[valid]>0).mean():.1f}%) = "
      f"{sign_test(int((f>0).sum()), len(f), p=float((-ret.loc[valid]>0).mean())):.4f}")
print(" ", cluster_note(epi, f, k=3))
order = np.argsort(-f)
for k in (1, 2, 3):
    keep = np.ones(len(f), bool); keep[order[:k]] = False
    print(f"  drop-best-{k}: {100*f[keep].mean():+.3f}%  (N={keep.sum()})")
yrs = pd.DatetimeIndex(epi).year
for lbl, m in [("pre-2018", yrs < 2018), ("2018+", yrs >= 2018),
               ("MIDTERM", (yrs % 4) == 2), ("non-midterm", (yrs % 4) != 2)]:
    if m.sum():
        print(f"  {lbl:<12} N={int(m.sum()):3d} mean {100*f[m].mean():+.3f}% "
              f"hit {100*(f[m]>0).mean():.1f}%")

print("\n  --- definition neighbours: threshold, lookback, and window ---")
rows = []
for thr in (90, 95, 98):
    for n in (10, 21, 42):
        c = cnt(CORE6, n=n, thr=thr)
        d, _ = cell(c >= 4, lbl=f"rank{n}>={thr}, count>=4")
        d["fade_mean_pct"] = -d.get("mean_pct", np.nan)
        rows.append(d)
show(rows, "count>=4 under 9 neighbouring definitions (LONG GDX h=5 episodes; flip sign for the fade)")

# ---------- 4. cost on the fade ----------
print("\n4. cost: GDX short round trip ~8 bps incl. borrow-free assumption.")
print(f"   fade episode mean {100*f.mean():.3f}% = {100*f.mean()*100:.1f} bps -> "
      f"{(100*f.mean()*100)/8.0:.1f}x cost (need >=5x); on the EXCESS over the")
print(f"   all-days short control it is {(100*(f.mean()-base)*100)/8.0:.1f}x.")

# ---------- 5. is today even in the population? ----------
print("\n5. today's position inside the trigger population")
trig = cal[live.values]
for t in ["GDX", "GLD"]:
    tr = _valid_pct_change(raw[t]["Close"].dropna(), 21).reindex(cal)
    q = float((tr.loc[trig] <= tr.iloc[-1]).mean())
    print(f"   {t} trailing 21d today {100*float(tr.iloc[-1]):+.2f}% sits at the "
          f"{100*q:.0f}th pctile of the trigger population "
          f"(median {100*float(tr.loc[trig].median()):+.2f}%)")
