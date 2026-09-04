"""b3 / C5: metals-complex thrust BREADTH.

PRE-DECLARED MEMBERSHIP, fixed before any forward return is measured:
  CORE6 = GLD, SLV, GDX, NEM, FCX, XME          (the map's complex)
  EXT9  = CORE6 + GDXJ, HG=F, SI=F              (the fuller complex)
Both are reported. Nothing else is added afterwards.

Trigger = count of members whose 21-day return sits at a trailing-252d PIT
rank >= 95, measured on each member's OWN valid sessions. Today CORE6 = 4.

The single biggest risk is the 2026-08-24 ENERGY COUNT trap: within the
11-name energy complex, long XLE h=5 by count of members at z10 >= 2.0 was
MONOTONE and crossed zero at four (2 names +0.715%, 3 +0.718%, 4 +0.139%,
5 -1.002%). If the metals count reproduces that shape -- broad thrust worse
than narrow -- and today's live count sits at or past the crossing, that is
a kill, not a finding.

Second required test (it killed the copper candidate on 2026-08-24): did the
METAL move, or only the equity? FCX ran +15.30% while HG=F was -0.30%.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change
import pandas as pd, numpy as np

pd.set_option("display.width", 240)

CORE6 = ["GLD", "SLV", "GDX", "NEM", "FCX", "XME"]
EXT3 = ["GDXJ", "HG=F", "SI=F"]
METALS = ["GC=F", "SI=F", "HG=F"]
VEH = ["GDX", "GLD"]

allt = sorted(set(CORE6 + EXT3 + METALS + VEH + ["SPY"]))
raw = load_prices(allt)

# per-ticker rank on its OWN valid sessions, then align (never a padded panel)
rk21 = {}
r21 = {}
for t in allt:
    s = raw[t]["Close"].dropna()
    rk21[t] = pct_rank(s, 21, 252)
    r21[t] = _valid_pct_change(s, 21)

# common trading calendar = GDX's (the vehicle); GLD/GDX both start 2006
cal = raw["GDX"]["Close"].dropna().index
px = pd.DataFrame({t: raw[t]["Close"] for t in allt}).reindex(cal).ffill(limit=1)
px = px.dropna(subset=VEH)
cal = px.index


def count_ge(members, thr=95.0):
    m = pd.DataFrame({t: (rk21[t] >= thr).reindex(cal).fillna(False) for t in members})
    ok = pd.DataFrame({t: rk21[t].reindex(cal).notna() for t in members})
    c = m.sum(axis=1)
    return c.where(ok.all(axis=1))


c6 = count_ge(CORE6)
c9 = count_ge(CORE6 + EXT3)
print(f"TODAY: CORE6 count = {c6.iloc[-1]:.0f}, EXT9 count = {c9.iloc[-1]:.0f}")
print("  member rank21 today:", {t: round(float(rk21[t].iloc[-1]), 1) for t in CORE6 + EXT3})
print("\ncount distribution (CORE6, days):")
print(c6.value_counts().sort_index().to_string())
print("\ncount distribution (EXT9, days):")
print(c9.value_counts().sort_index().to_string())

# ---------------- THE ENERGY-COUNT TRAP: monotonicity by count ----------------
print("\n" + "=" * 78)
print("MONOTONICITY BY COUNT -- the 2026-08-24 energy trap, run head on")
print("=" * 78)
for veh in VEH:
    for h in (3, 5, 10):
        ret = vehicle_ret(px, [(veh, 1.0)], h, 1)
        valid = ret.dropna().index
        rows = []
        for k in range(0, 7):
            tt = cal[(c6 == k).reindex(cal, fill_value=False).values].intersection(valid)
            epi = declusters(tt, h, valid)
            d = summarize(ret.loc[epi].values, f"count=={k}")
            d["n_days"] = len(tt)
            rows.append(d)
        base = summarize(ret.loc[valid].values, "ALL DAYS")
        rows.append(base)
        show(rows, f"CORE6 count -> LONG {veh} h={h} (episodes)")

# cumulative form: >= k
print("\n--- cumulative form, LONG GDX h=5 ---")
ret = vehicle_ret(px, [("GDX", 1.0)], 5, 1)
valid = ret.dropna().index
rows = []
for k in range(1, 7):
    tt = cal[(c6 >= k).reindex(cal, fill_value=False).values].intersection(valid)
    epi = declusters(tt, 5, valid)
    d = summarize(ret.loc[epi].values, f"count>={k}")
    d["n_days"] = len(tt)
    rows.append(d)
rows.append(summarize(ret.loc[valid].values, "ALL DAYS"))
show(rows)

print("\n--- cumulative form, LONG GLD h=5 ---")
retg = vehicle_ret(px, [("GLD", 1.0)], 5, 1)
validg = retg.dropna().index
rows = []
for k in range(1, 7):
    tt = cal[(c6 >= k).reindex(cal, fill_value=False).values].intersection(validg)
    epi = declusters(tt, 5, validg)
    d = summarize(retg.loc[epi].values, f"count>={k}")
    d["n_days"] = len(tt)
    rows.append(d)
rows.append(summarize(retg.loc[validg].values, "ALL DAYS"))
show(rows)

# ---------------- battery at today's live count ----------------
live_mask = (c6 >= 4).reindex(cal, fill_value=False)
variants = {f"CORE6 count>={k}": (c6 >= k).reindex(cal, fill_value=False) for k in (2, 3, 4, 5)}
variants["EXT9 count>=4"] = (c9 >= 4).reindex(cal, fill_value=False)
variants["EXT9 count>=6"] = (c9 >= 6).reindex(cal, fill_value=False)
battery(px, live_mask, [("GDX", 1.0)], 5, "C5 LONG GDX on CORE6 count>=4 (today's live count)",
        8.0, variants=variants, event_kinds=("cpi", "fomc_decision"))
battery(px, live_mask, [("GLD", 1.0)], 5, "C5 LONG GLD on CORE6 count>=4",
        4.0, variants=variants, event_kinds=("cpi", "fomc_decision"))

# ---------------- DID THE METAL MOVE? ----------------
print("\n" + "=" * 78)
print("DID THE METAL MOVE, OR ONLY THE EQUITY? (the copper kill)")
print("=" * 78)
trig = cal[live_mask.values]
print(f"count>=4 trigger days: {len(trig)}")
rows = []
for t in ["GC=F", "SI=F", "HG=F", "GLD", "SLV", "GDX", "NEM", "FCX", "XME"]:
    tr = r21[t].reindex(cal)
    rows.append({"ticker": t,
                 "median_trailing_r21_on_trig_pct": round(100 * float(tr.loc[trig].median()), 2),
                 "median_all_days_pct": round(100 * float(tr.median()), 2),
                 "today_pct": round(100 * float(tr.iloc[-1]), 2)})
print(pd.DataFrame(rows).to_string(index=False))

# equity-only vs metal-only breadth
EQ = ["GDX", "NEM", "FCX", "XME"]
MET = ["GLD", "SLV"]
ceq = count_ge(EQ)
cmet = count_ge(MET)
print(f"\nTODAY equity-leg count (GDX/NEM/FCX/XME >=95) = {ceq.iloc[-1]:.0f} of 4; "
      f"metal-leg count (GLD/SLV) = {cmet.iloc[-1]:.0f} of 2")
rows = []
for lbl, m in [("equity count>=3", ceq >= 3), ("equity count>=3 & metal count==0", (ceq >= 3) & (cmet == 0)),
               ("equity count>=3 & metal count>=1", (ceq >= 3) & (cmet >= 1)),
               ("metal count>=1 alone", cmet >= 1)]:
    tt = cal[m.reindex(cal, fill_value=False).values].intersection(valid)
    epi = declusters(tt, 5, valid)
    d = summarize(ret.loc[epi].values, lbl)
    d["n_days"] = len(tt)
    rows.append(d)
rows.append(summarize(ret.loc[valid].values, "ALL DAYS"))
show(rows, "LONG GDX h=5 by which LEG carries the breadth")

# ---------------- trailing return on trigger days ----------------
print("\n=== trailing 21d return of the VEHICLE on trigger days (lagging-marker check) ===")
for veh in VEH:
    tr = r21[veh].reindex(cal)
    print(f"  {veh}: median trailing 21d on count>=4 days = {100*float(tr.loc[trig].median()):+.2f}% "
          f"(all days {100*float(tr.median()):+.2f}%); today {100*float(tr.iloc[-1]):+.2f}%")
